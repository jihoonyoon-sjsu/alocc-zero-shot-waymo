"""Read trusted local ALOcc prediction pickles one sample at a time.

This is separate from the JSON file connector. It understands two saved
layouts and does not treat them as the same schema:

- nuScenes: a list of dictionaries with ``pred_occupancy`` and ``index``
- Waymo: a list of dictionaries with ``scene_idx``, ``frame_idx``, and
  ``pred_occupancy``, or one dictionary with those keys

Missing flow stays missing. The motion-flow head is outside the current
project scope, so this module does not invent flow, turn a 2-component flow
array into a 3-component prediction, or publish a zero flow field.

When the occupancy array shape matches ``confirmed_alocc``, class names and
ego-frame voxel geometry come from that configuration. Other shapes keep
provisional index labels and do not claim a meter origin. Sensor timestamps
are not invented. A stamp of 0 with ``timestamp_available`` false is not a
capture time. Waymo ``scene_idx`` and ``frame_idx`` are copied from the
pickle and are not matched to an external frame archive here.
"""

from __future__ import annotations

import argparse
import json
import pickle
import queue
import threading
from pathlib import Path

from alocc_ros.connectors.base import ConnectorUnavailable, PredictionConnector
from alocc_ros.occupancy_geometry import spatial_from_confirmed
from alocc_ros.prediction_types import (
    FlowPrediction,
    OccupancyPrediction,
    SpatialMetadata,
)

PICKLE_MAX_DIMENSION = 256
AXIS_CONVENTION = "PICKLE_AXIS0_FASTEST_AXIS1_THEN_AXIS2_GEOMETRY_UNCONFIRMED"
UNCONFIRMED_FLOW_UNITS = "UNCONFIRMED_NOT_A_PHYSICAL_UNIT"
_END = object()


class _StopStreaming(Exception):
    """Stop reading a pickle without treating that as a corrupt file."""


class _DropList(list):
    """Top-level pickle list that yields each item and does not retain it."""

    def __init__(self, output: queue.Queue, stop: threading.Event):
        super().__init__()
        self._output = output
        self._stop = stop

    def append(self, item) -> None:
        _emit(self._output, self._stop, item)

    def extend(self, items) -> None:
        for item in items:
            _emit(self._output, self._stop, item)


def _emit(output: queue.Queue, stop: threading.Event, item) -> None:
    while not stop.is_set():
        try:
            output.put(item, timeout=0.2)
            return
        except queue.Full:
            continue
    raise _StopStreaming


class _StreamingUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, file, output: queue.Queue, stop: threading.Event):
        super().__init__(file)
        self._output = output
        self._stop = stop
        self.replaced_root = False
        # True after a list item was already handed off during SETITEMS.
        # A following one-item APPEND must not pop the list itself.
        self._released_list_item = False

    def load_empty_list(self):
        if not self.replaced_root and not self.stack:
            self.replaced_root = True
            self.append(_DropList(self._output, self._stop))
        else:
            pickle._Unpickler.load_empty_list(self)

    def load_setitems(self):
        pickle._Unpickler.load_setitems(self)
        self._release_finished_sample()

    def load_append(self):
        if self._released_list_item and self.stack and isinstance(self.stack[-1], _DropList):
            self._released_list_item = False
            return
        value = self.stack.pop()
        parent = self.stack[-1]
        if isinstance(parent, _DropList):
            self._hand_off(value)
            return
        list.append(parent, value)

    def load_appends(self):
        items = self.pop_mark()
        parent = self.stack[-1]
        if isinstance(parent, _DropList):
            # Samples already popped in SETITEMS are not in items.
            self._released_list_item = False
            for item in items:
                self._hand_off(item)
            return
        try:
            extend = parent.extend
        except AttributeError:
            for item in items:
                parent.append(item)
        else:
            extend(items)

    def _release_finished_sample(self) -> None:
        """Yield one finished sample before the next pickle batch accumulates.

        Python 3.12's pure-Python unpickler implements MARK by swapping
        ``stack`` onto ``metastack`` and starting a fresh list. A protocol-2
        APPENDS batch therefore collects dicts on that fresh list, while the
        top-level list sits one frame down. Pop the finished dict here so the
        batch cannot retain every array until APPENDS.
        """
        if not self.stack or not self.metastack or not isinstance(self.stack[-1], dict):
            return
        parent_frame = self.metastack[-1]
        if not parent_frame or not isinstance(parent_frame[-1], _DropList):
            return
        item = self.stack.pop()
        self._released_list_item = True
        self._hand_off(item)

    def _hand_off(self, item) -> None:
        if isinstance(item, dict):
            _forget_large_memos(self.memo, item)
        _emit(self._output, self._stop, item)


def _forget_large_memos(memo: dict, sample: dict) -> None:
    """Drop this sample's large memo entries so earlier frames can be freed.

    Later samples BINGET shared functions, dtypes, short strings, and small
    arrays. Those stay. The sample dict, its occupancy array, and the large
    pickle strings that built that array do not.
    """
    for key, value in list(memo.items()):
        if key == 0:
            continue
        if value is sample or _is_large_memo(value):
            # Keep the key. Protocol 4 MEMOIZE uses len(memo) as the next id.
            memo[key] = None


def _is_large_memo(value) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) > 256
    if isinstance(value, tuple):
        return any(_is_large_memo(item) for item in value)
    if isinstance(value, list):
        return any(_is_large_memo(item) for item in value)
    numpy = _numpy_or_none()
    if numpy is not None and isinstance(value, numpy.ndarray):
        return int(value.nbytes) > 4096
    return False


def _numpy_or_none():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


_StreamingUnpickler.dispatch[ord("]")] = _StreamingUnpickler.load_empty_list
_StreamingUnpickler.dispatch[ord("u")] = _StreamingUnpickler.load_setitems
_StreamingUnpickler.dispatch[ord("a")] = _StreamingUnpickler.load_append
_StreamingUnpickler.dispatch[ord("e")] = _StreamingUnpickler.load_appends


class _SampleStream:
    """Background unpickle of one trusted file. Items are handed off and dropped."""

    def __init__(self, path: Path):
        self._path = path
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.root_kind = ""
        self._thread = threading.Thread(target=self._run, name="alocc-pkl-stream", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def __iter__(self):
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _run(self) -> None:
        handle = None
        try:
            handle = self._path.open("rb")
            unpickler = _StreamingUnpickler(handle, self._queue, self._stop)
            result = unpickler.load()
            if unpickler.replaced_root:
                self.root_kind = "list"
            elif isinstance(result, dict):
                self.root_kind = "dict"
                _emit(self._queue, self._stop, result)
            elif isinstance(result, list):
                self.root_kind = "list"
                for item in result:
                    _emit(self._queue, self._stop, item)
            else:
                self.root_kind = type(result).__name__
                raise ConnectorUnavailable(
                    f"{self._path} root is {type(result).__name__}, not a list or dictionary"
                )
        except _StopStreaming:
            pass
        except Exception as exc:
            if not self._stop.is_set():
                self._queue.put(exc)
        finally:
            if handle is not None:
                handle.close()
            self._queue.put(_END)


class PicklePredictionConnector(PredictionConnector):
    name = "pkl"

    def __init__(self, model_config: dict):
        self._model_config = model_config
        pkl_cfg = model_config.get("pkl") or {}
        self._path = Path(str(pkl_cfg.get("path") or "").strip())
        self._loop = bool(pkl_cfg.get("loop", False))
        self._max_samples = int(pkl_cfg.get("max_samples", 0))
        self._stream: _SampleStream | None = None
        self._iterator = None
        self._position = 0
        self._loaded = 0
        self._success_this_pass = 0
        self._skip_records: list[dict[str, str]] = []
        self._expected_shape: tuple[int, int, int] | None = None

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._iterator = None

    def drain_skip_records(self) -> list[dict[str, str]]:
        records = self._skip_records
        self._skip_records = []
        return records

    def load(self) -> tuple[OccupancyPrediction, FlowPrediction | None]:
        if not str(self._path):
            raise ConnectorUnavailable("Pickle connector needs pkl.path.")
        if self._max_samples and self._loaded >= self._max_samples:
            raise ConnectorUnavailable(
                f"pkl.max_samples={self._max_samples} reached for {self._path.name}"
            )
        if self._iterator is None:
            self._open()
        while True:
            try:
                sample = next(self._iterator)
            except StopIteration:
                if self._success_this_pass == 0 or not self._loop:
                    raise ConnectorUnavailable(f"No more prediction samples in {self._path}") from None
                self.close()
                self._position = 0
                self._success_this_pass = 0
                self._expected_shape = None
                self._open()
                continue
            position = self._position
            self._position += 1
            try:
                occupancy, flow = convert_saved_sample(sample, position, self._model_config)
                if self._expected_shape is None:
                    self._expected_shape = (
                        occupancy.spatial.dim_x,
                        occupancy.spatial.dim_y,
                        occupancy.spatial.dim_z,
                    )
                elif (
                    occupancy.spatial.dim_x,
                    occupancy.spatial.dim_y,
                    occupancy.spatial.dim_z,
                ) != self._expected_shape:
                    raise ConnectorUnavailable(
                        f"sample {position} shape "
                        f"{(occupancy.spatial.dim_x, occupancy.spatial.dim_y, occupancy.spatial.dim_z)} "
                        f"does not match the first sample {self._expected_shape}"
                    )
            except ConnectorUnavailable as exc:
                self._skip_records.append(
                    {"sample_index": str(position), "reason": str(exc)}
                )
                continue
            self._loaded += 1
            self._success_this_pass += 1
            return occupancy, flow

    def _open(self) -> None:
        if not self._path.is_file():
            raise ConnectorUnavailable(f"Prediction pickle not found: {self._path}")
        self._stream = _SampleStream(self._path)
        self._iterator = iter(self._stream)


def convert_saved_sample(
    sample, sample_index: int, model_config: dict | None = None
) -> tuple[OccupancyPrediction, FlowPrediction | None]:
    if not isinstance(sample, dict):
        raise ConnectorUnavailable(
            f"sample {sample_index} is {type(sample).__name__}, not a dictionary"
        )
    if "scene_idx" in sample and "frame_idx" in sample:
        return _convert_waymo(sample, sample_index, model_config)
    if "pred_occupancy" in sample and ("index" in sample or "pred_flow" in sample):
        return _convert_nuscenes(sample, sample_index, model_config)
    if "pred_occupancy" in sample:
        raise ConnectorUnavailable(
            f"sample {sample_index} has pred_occupancy but neither Waymo "
            "scene_idx/frame_idx nor nuScenes index/pred_flow keys"
        )
    raise ConnectorUnavailable(f"sample {sample_index} has no pred_occupancy")


def _convert_nuscenes(sample: dict, sample_index: int, model_config: dict | None):
    confirmed = _shape_matches_confirmed(sample.get("pred_occupancy"), model_config)
    occupancy = _occupancy_from_array(
        sample.get("pred_occupancy"),
        sample_index,
        _metadata(
            "nuscenes",
            sample,
            ("index", "pts_bbox", "iou", "pred_density"),
            class_names_confirmed=confirmed,
        ),
        _flow_was_supplied(sample),
        model_config,
    )
    flow = _flow_from_value(sample.get("pred_flow"), occupancy, "nuscenes")
    occupancy = _with_flow_flag(occupancy, flow is not None)
    return occupancy, flow


def _convert_waymo(sample: dict, sample_index: int, model_config: dict | None):
    confirmed = _shape_matches_confirmed(sample.get("pred_occupancy"), model_config)
    occupancy = _occupancy_from_array(
        sample.get("pred_occupancy"),
        sample_index,
        _metadata(
            "waymo",
            sample,
            ("scene_idx", "frame_idx"),
            class_names_confirmed=confirmed,
        ),
        "pred_flow" in sample and _flow_value_usable(sample.get("pred_flow")),
        model_config,
    )
    if "pred_flow" not in sample:
        return occupancy, None
    flow = _flow_from_value(sample.get("pred_flow"), occupancy, "waymo")
    occupancy = _with_flow_flag(occupancy, flow is not None)
    return occupancy, flow


def _with_flow_flag(occupancy: OccupancyPrediction, supplied: bool) -> OccupancyPrediction:
    if occupancy.flow_supplied == supplied:
        return occupancy
    return OccupancyPrediction(
        spatial=occupancy.spatial,
        class_names=occupancy.class_names,
        semantic_labels=occupancy.semantic_labels,
        timestamp_sec=occupancy.timestamp_sec,
        timestamp_nanosec=occupancy.timestamp_nanosec,
        is_mock=occupancy.is_mock,
        source=occupancy.source,
        label_layout=occupancy.label_layout,
        timestamp_available=occupancy.timestamp_available,
        sample_index=occupancy.sample_index,
        sample_metadata=occupancy.sample_metadata,
        flow_supplied=supplied,
    )


def _occupancy_from_array(
    array,
    sample_index: int,
    metadata: str,
    flow_supplied: bool,
    model_config: dict | None = None,
):
    values, shape = _occupancy_values(array, sample_index)
    spatial, class_names = _spatial_and_classes(values, shape, sample_index, model_config)
    flat = values.ravel(order="F")
    occupancy = OccupancyPrediction(
        spatial=spatial,
        class_names=class_names,
        semantic_labels=tuple(int(label) for label in flat.tolist()),
        timestamp_sec=0,
        timestamp_nanosec=0,
        is_mock=False,
        source="pkl",
        timestamp_available=False,
        sample_index=sample_index,
        sample_metadata=metadata,
        flow_supplied=flow_supplied,
    )
    occupancy.validate()
    return occupancy


def _confirmed_section(model_config: dict | None) -> dict | None:
    if not model_config:
        return None
    section = model_config.get("confirmed_alocc")
    if not isinstance(section, dict):
        return None
    return section


def _shape_matches_confirmed(array, model_config: dict | None) -> bool:
    section = _confirmed_section(model_config)
    numpy = _numpy_or_none()
    if section is None or numpy is None or not isinstance(array, numpy.ndarray) or array.ndim != 3:
        return False
    shape = tuple(int(dim) for dim in array.shape)
    expected = (int(section["dim_x"]), int(section["dim_y"]), int(section["dim_z"]))
    return shape == expected


def _spatial_and_classes(values, shape, sample_index: int, model_config: dict | None):
    """Apply the confirmed ego grid only when the saved array has that shape.

    A smaller or larger array keeps provisional labels. Applying the 200 x
    200 x 16 meter origin to a different shape would misplace every voxel.
    """
    section = _confirmed_section(model_config)
    expected = None
    if section is not None:
        expected = (int(section["dim_x"]), int(section["dim_y"]), int(section["dim_z"]))
    if section is not None and shape == expected:
        class_names = tuple(str(name) for name in section["class_names"])
        lowest = int(values.min())
        highest = int(values.max())
        if lowest < 0 or highest >= len(class_names):
            raise ConnectorUnavailable(
                f"sample {sample_index} occupancy labels {lowest}..{highest} fall outside "
                f"confirmed class ids 0..{len(class_names) - 1}"
            )
        return spatial_from_confirmed(section), class_names
    highest = int(values.max())
    class_names = tuple(f"unconfirmed_label_{label}" for label in range(highest + 1))
    spatial = SpatialMetadata(
        dim_x=shape[0],
        dim_y=shape[1],
        dim_z=shape[2],
        resolution_x=float("nan"),
        resolution_y=float("nan"),
        resolution_z=float("nan"),
        origin_x=float("nan"),
        origin_y=float("nan"),
        origin_z=float("nan"),
        coordinate_convention=AXIS_CONVENTION,
        geometry_confirmed=False,
        max_dimension=PICKLE_MAX_DIMENSION,
    )
    return spatial, class_names


def _occupancy_values(array, sample_index: int):
    numpy = _numpy()
    if not isinstance(array, numpy.ndarray):
        raise ConnectorUnavailable(
            f"sample {sample_index} pred_occupancy is {type(array).__name__}, not a numpy array"
        )
    if array.ndim != 3:
        raise ConnectorUnavailable(
            f"sample {sample_index} pred_occupancy ndim is {array.ndim}, expected 3"
        )
    if array.dtype != numpy.uint8:
        raise ConnectorUnavailable(
            f"sample {sample_index} pred_occupancy dtype is {array.dtype}, expected uint8"
        )
    if array.size == 0:
        raise ConnectorUnavailable(f"sample {sample_index} pred_occupancy is empty")
    shape = tuple(int(dim) for dim in array.shape)
    return array, shape


def _flow_from_value(value, occupancy: OccupancyPrediction, dataset: str):
    if value is None:
        return None
    numpy = _numpy()
    if not isinstance(value, numpy.ndarray):
        raise ConnectorUnavailable(
            f"{dataset} pred_flow is {type(value).__name__}, not an array or null"
        )
    if value.shape[-1] != 3 or value.ndim != 4:
        # Keep occupancy. Do not pad a missing component and call it flow.
        return None
    if not numpy.isfinite(value).all():
        return None
    expected = (
        occupancy.spatial.dim_x,
        occupancy.spatial.dim_y,
        occupancy.spatial.dim_z,
        3,
    )
    if tuple(int(dim) for dim in value.shape) != expected:
        return None
    ordered = []
    axis0, axis1, axis2, _channels = value.shape
    for k in range(axis2):
        for j in range(axis1):
            for i in range(axis0):
                ordered.extend(
                    (float(value[i, j, k, 0]), float(value[i, j, k, 1]), float(value[i, j, k, 2]))
                )
    flow = FlowPrediction(
        spatial=occupancy.spatial,
        flow_vectors=tuple(ordered),
        timestamp_sec=0,
        timestamp_nanosec=0,
        is_mock=False,
        source="pkl",
        flow_units=UNCONFIRMED_FLOW_UNITS,
    )
    flow.validate()
    return flow


def _flow_was_supplied(sample: dict) -> bool:
    return _flow_value_usable(sample.get("pred_flow"))


def _flow_value_usable(value) -> bool:
    numpy = _numpy()
    return isinstance(value, numpy.ndarray) and value.ndim == 4 and value.shape[-1] == 3


def _metadata(
    dataset: str,
    sample: dict,
    keys: tuple[str, ...],
    class_names_confirmed: bool = False,
) -> str:
    class_label = (
        "class_names=nuscenes_semantic_18"
        if class_names_confirmed
        else "class_names=provisional_index_labels"
    )
    parts = [
        f"dataset={dataset}",
        class_label,
        "flow_units_confirmed=false",
        "timestamp=unavailable",
    ]
    for key in keys:
        if key not in sample:
            parts.append(f"{key}=absent")
            continue
        value = sample[key]
        if value is None:
            parts.append(f"{key}=null")
        elif isinstance(value, (str, int, bool)):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={type(value).__name__}")
    if "pred_flow" not in sample:
        parts.append("pred_flow=absent")
    elif sample.get("pred_flow") is None:
        parts.append("pred_flow=null")
    elif _flow_value_usable(sample.get("pred_flow")):
        parts.append("pred_flow=usable")
    else:
        parts.append("pred_flow=present_but_not_usable")
    return ";".join(parts)


def _rss_mb() -> float | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return None


def _numpy():
    try:
        import numpy
    except ImportError as exc:
        raise ConnectorUnavailable("Reading an ALOcc pickle needs numpy.") from exc
    return numpy


def summarize_pickle(path: str | Path) -> dict:
    """Scan a trusted pickle and return shapes, ranges, and flow presence.

    Arrays are discarded after each sample. The returned dictionary does not
    contain voxel values.
    """
    file_path = Path(path)
    numpy = _numpy()
    stream = _SampleStream(file_path)
    summary = {
        "path": str(file_path),
        "bytes": file_path.stat().st_size,
        "sample_count": 0,
        "loaded_count": 0,
        "skipped_count": 0,
        "skip_reasons": [],
        "dataset": "",
        "keys": [],
        "occupancy_shape": None,
        "occupancy_dtype": "",
        "shapes_consistent": True,
        "label_min": None,
        "label_max": None,
        "label_ids_present": [],
        "flow_field_present": False,
        "flow_usable_count": 0,
        "flow_null_count": 0,
        "flow_absent_count": 0,
        "flow_other_count": 0,
        "timestamp_field_present": False,
        "metadata_extrema": {},
        "root_kind": "",
    }
    counts = numpy.zeros(256, dtype=numpy.int64)
    extrema = {}
    try:
        for position, sample in enumerate(stream):
            summary["sample_count"] += 1
            if summary["sample_count"] % 500 == 0:
                rss = _rss_mb()
                rss_text = "unknown" if rss is None else f"{rss:.1f} MB"
                print(
                    f"scanned {summary['sample_count']} samples, rss {rss_text}",
                    flush=True,
                )
            if not isinstance(sample, dict):
                summary["skipped_count"] += 1
                summary["skip_reasons"].append(f"{position}: {type(sample).__name__}")
                continue
            if not summary["keys"]:
                summary["keys"] = sorted(str(key) for key in sample.keys())
                summary["dataset"] = _dataset_name(sample)
                summary["timestamp_field_present"] = any(
                    key in sample for key in ("timestamp", "timestamp_sec", "token", "sample_token")
                )
            if "pred_flow" not in sample:
                summary["flow_absent_count"] += 1
            elif sample["pred_flow"] is None:
                summary["flow_field_present"] = True
                summary["flow_null_count"] += 1
            elif _flow_value_usable(sample["pred_flow"]):
                summary["flow_field_present"] = True
                summary["flow_usable_count"] += 1
            else:
                summary["flow_field_present"] = True
                summary["flow_other_count"] += 1
            try:
                values, shape = _occupancy_values(sample.get("pred_occupancy"), position)
            except ConnectorUnavailable as exc:
                summary["skipped_count"] += 1
                if len(summary["skip_reasons"]) < 20:
                    summary["skip_reasons"].append(str(exc))
                continue
            if summary["occupancy_shape"] is None:
                summary["occupancy_shape"] = list(shape)
                summary["occupancy_dtype"] = str(values.dtype)
            elif list(shape) != summary["occupancy_shape"] or str(values.dtype) != summary["occupancy_dtype"]:
                summary["shapes_consistent"] = False
            counts += numpy.bincount(values.ravel(), minlength=256)
            summary["loaded_count"] += 1
            for key in ("index", "scene_idx", "frame_idx"):
                value = sample.get(key)
                if isinstance(value, (int, numpy.integer)):
                    number = int(value)
                    lowest, highest = extrema.get(key, (number, number))
                    extrema[key] = (min(lowest, number), max(highest, number))
            del values
    finally:
        stream.close()
    present = numpy.flatnonzero(counts)
    if present.size:
        summary["label_min"] = int(present[0])
        summary["label_max"] = int(present[-1])
        summary["label_ids_present"] = [int(label) for label in present.tolist()]
    summary["metadata_extrema"] = {
        key: {"min": bounds[0], "max": bounds[1]} for key, bounds in extrema.items()
    }
    summary["root_kind"] = stream.root_kind or ("empty" if summary["sample_count"] == 0 else "unknown")
    return summary


def _dataset_name(sample: dict) -> str:
    if "scene_idx" in sample and "frame_idx" in sample:
        return "waymo"
    if "index" in sample or "pred_flow" in sample:
        return "nuscenes"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize a trusted ALOcc prediction pickle.")
    parser.add_argument("path")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    summary = summarize_pickle(args.path)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
