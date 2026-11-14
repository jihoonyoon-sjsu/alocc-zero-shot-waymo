#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="mock"
MODEL_CONFIG="${ROOT}/config/model.yaml"
ROS_CONFIG="${ROOT}/config/ros2.yaml"
INPUT=""
MAX_SAMPLES="8"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model-config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    --ros-config)
      ROS_CONFIG="$2"
      shift 2
      ;;
    --input)
      INPUT="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 --mode mock [--model-config PATH] [--ros-config PATH]" >&2
      echo "       $0 --mode pkl --input PICKLE [--max-samples N]" >&2
      exit 1
      ;;
  esac
done

if [ ! -f "${ROOT}/install/setup.bash" ]; then
  echo "The project is not built. Run ./scripts/build.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
set +u
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source "${ROOT}/install/setup.bash"
set -u
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

if [ "${MODE}" = "pkl" ]; then
  if [ -z "${INPUT}" ]; then
    echo "pkl mode needs --input PATH to a trusted ALOcc prediction pickle." >&2
    exit 1
  fi
  if [ ! -f "${INPUT}" ]; then
    echo "Prediction pickle not found: ${INPUT}" >&2
    exit 1
  fi
  STEM="$(basename "${INPUT}" .pkl)"
  RESULT_DIR="${ROOT}/results/real_occupancy/${STEM}"
  mkdir -p "${RESULT_DIR}/runtime" "${RESULT_DIR}/visualization"
  rm -f "${RESULT_DIR}/runtime/"*.csv "${RESULT_DIR}/visualization/"*.svg \
    "${RESULT_DIR}/visualization/marker_summary.txt" "${RESULT_DIR}/inspection.json"
  WORK="$(mktemp -d)"
  trap 'rm -rf "${WORK}"' EXIT
  python3 - "${MODEL_CONFIG}" "${INPUT}" "${MAX_SAMPLES}" "${WORK}/model.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    model = yaml.safe_load(handle)
model.setdefault("pkl", {})
model["pkl"]["path"] = sys.argv[2]
model["pkl"]["loop"] = False
model["pkl"]["max_samples"] = int(sys.argv[3])
with open(sys.argv[4], "w", encoding="utf-8") as handle:
    yaml.safe_dump(model, handle)
PY
  python3 - "${ROS_CONFIG}" "${WORK}/ros2.yaml" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["publish_rate_hz"] = 1.0
config["demo_duration_sec"] = 30
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle)
PY
  python3 -m alocc_ros.config_loader \
    --mode pkl \
    --model-config "${WORK}/model.yaml" \
    --ros-config "${WORK}/ros2.yaml"
  echo "Scanning ${INPUT} before publishing."
  python3 -m alocc_ros.connectors.pickle_input "${INPUT}" --output "${RESULT_DIR}/inspection.json"

  set +e
  timeout --foreground -s INT 240 ros2 launch alocc_ros mock_demo.launch.py \
    mode:=pkl \
    model_config:="${WORK}/model.yaml" \
    ros_config:="${WORK}/ros2.yaml" \
    results_dir:="${RESULT_DIR}"
  status=$?
  set -e
  if [ "${status}" -ne 0 ] && [ "${status}" -ne 124 ] && [ "${status}" -ne 130 ]; then
    echo "Pickle demo launch failed with status ${status}." >&2
    exit "${status}"
  fi

  python3 - "${RESULT_DIR}" "${MAX_SAMPLES}" <<'PY'
import csv
import json
import sys
from pathlib import Path

result = Path(sys.argv[1])
expected = int(sys.argv[2])
runtime = result / "runtime"
inspection = json.loads((result / "inspection.json").read_text(encoding="utf-8"))
published = list(csv.DictReader((runtime / "published_occupancy.csv").open(encoding="utf-8")))
received = list(csv.DictReader((runtime / "received_occupancy.csv").open(encoding="utf-8")))
validation = list(csv.DictReader((runtime / "validation.csv").open(encoding="utf-8")))
summary = (runtime / "summary.csv").read_text(encoding="utf-8")
flow_status = (runtime / "flow_status.csv").read_text(encoding="utf-8")
occupancy_svg = (result / "visualization" / "occupancy_slice.svg").read_text(encoding="utf-8")
marker_summary = (result / "visualization" / "marker_summary.txt").read_text(encoding="utf-8")

if "real_occupancy_pickle_not_model_inference" not in summary:
    raise SystemExit("Summary is missing the real occupancy measurement context.")
if "flow was not supplied" not in flow_status:
    raise SystemExit("Flow status does not say that flow was not supplied.")
if "REAL occupancy" not in occupancy_svg or "geometry=ego_frame" not in occupancy_svg:
    raise SystemExit("Occupancy slice is not labelled as real ego-frame data.")
if "geometry=unconfirmed_voxel_indices" in occupancy_svg:
    raise SystemExit("Real occupancy slice still says geometry is unconfirmed.")
if "frame=ego" not in marker_summary or "truncated=false" not in marker_summary:
    raise SystemExit("Marker summary did not record a full-resolution ego-frame draw.")
if "skipped_free=" not in marker_summary:
    raise SystemExit("Marker summary did not record free-space filtering.")
skipped_free = 0
rendered = 0
for token in marker_summary.split():
    if token.startswith("skipped_free="):
        skipped_free = int(token.split("=", 1)[1])
    if token.startswith("rendered="):
        rendered = int(token.split("=", 1)[1])
if rendered < 1 or skipped_free < 1:
    raise SystemExit(
        f"Expected occupied cubes and omitted free voxels, got rendered={rendered} "
        f"skipped_free={skipped_free}."
    )
if (result / "visualization" / "flow_slice.svg").exists():
    raise SystemExit("A flow slice was written even though flow was not supplied.")
occupancy_validation = [row for row in validation if row["kind"] == "occupancy"]
if len(published) < 1:
    raise SystemExit("No occupancy samples were published.")
if len(published) > expected:
    raise SystemExit(f"Published {len(published)} samples, above the cap of {expected}.")
if any(row["passed"] != "true" for row in occupancy_validation):
    raise SystemExit("Occupancy validation failed.")
if any(row["kind"] == "flow" for row in validation):
    raise SystemExit("A flow message was validated even though flow was not supplied.")
by_index = {row["sample_index"]: row for row in received}
for row in published:
    match = by_index.get(row["sample_index"])
    if match is None:
        raise SystemExit(f"Subscriber missed sample {row['sample_index']}.")
    if match["label_sha256"] != row["label_sha256"]:
        raise SystemExit(f"Received labels do not match sample {row['sample_index']}.")
    if match["dim_x"] != row["dim_x"] or match["dim_y"] != row["dim_y"] or match["dim_z"] != row["dim_z"]:
        raise SystemExit(f"Received dimensions do not match sample {row['sample_index']}.")
    if match["flow_supplied"] != "false" or row["flow_supplied"] != "false":
        raise SystemExit("Missing flow was marked as supplied.")
    if match["is_mock"] != "false":
        raise SystemExit("Real occupancy was marked as mock.")
    if row["dim_x"] != "200" or row["dim_y"] != "200" or row["dim_z"] != "16":
        raise SystemExit(f"Sample {row['sample_index']} is not the confirmed 200x200x16 grid.")
    if "class_names=nuscenes_semantic_18" not in row["sample_metadata"]:
        raise SystemExit(f"Sample {row['sample_index']} is missing the confirmed class mapping.")
    if "timestamp=unavailable" not in row["sample_metadata"]:
        raise SystemExit(f"Sample {row['sample_index']} does not mark its timestamp unavailable.")
print(
    "Real occupancy pickle demo completed. "
    f"Found {inspection['sample_count']} samples, published {len(published)}, "
    f"received {len(by_index)}. Results are in {result}."
)
PY
  exit 0
fi

python3 -m alocc_ros.config_loader \
  --mode "${MODE}" \
  --model-config "${MODEL_CONFIG}" \
  --ros-config "${ROS_CONFIG}"

mkdir -p "${ROOT}/results/runtime" "${ROOT}/results/visualization"
rm -f "${ROOT}/results/runtime/"*.csv "${ROOT}/results/visualization/"*.svg

DURATION="$(python3 - "${ROS_CONFIG}" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
print(int(float(config["demo_duration_sec"])) + 12)
PY
)"

set +e
timeout --foreground -s INT "${DURATION}" ros2 launch alocc_ros mock_demo.launch.py \
  mode:="${MODE}" \
  model_config:="${MODEL_CONFIG}" \
  ros_config:="${ROS_CONFIG}" \
  results_dir:="${ROOT}/results"
status=$?
set -e

if [ "${status}" -ne 0 ] && [ "${status}" -ne 124 ] && [ "${status}" -ne 130 ]; then
  echo "Demo launch failed with status ${status}." >&2
  exit "${status}"
fi

python3 - "${ROOT}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
required = [
    root / "results/runtime/stage_timings.csv",
    root / "results/runtime/latency.csv",
    root / "results/runtime/summary.csv",
    root / "results/runtime/received_occupancy.csv",
    root / "results/runtime/received_flow.csv",
    root / "results/runtime/validation.csv",
    root / "results/visualization/occupancy_slice.svg",
    root / "results/visualization/flow_slice.svg",
]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit("Demo outputs are missing or empty:\n" + "\n".join(missing))

summary = (root / "results/runtime/summary.csv").read_text(encoding="utf-8")
if "mock_pipeline_not_model_inference" not in summary:
    raise SystemExit("Summary is missing the mock measurement context.")
if "p95_occupancy_e2e_latency_s" not in summary:
    raise SystemExit("Summary is missing p95 latency.")

with (root / "results/runtime/validation.csv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows or any(row["passed"] != "true" for row in rows):
    raise SystemExit("Prediction validation did not pass for every received sample.")

occupancy_svg = (root / "results/visualization/occupancy_slice.svg").read_text(encoding="utf-8")
flow_svg = (root / "results/visualization/flow_slice.svg").read_text(encoding="utf-8")
if "MOCK occupancy" not in occupancy_svg or "MOCK flow" not in flow_svg:
    raise SystemExit("Visualization files are not labelled as mock data.")
print("Mock demo completed. Results are in results/runtime and results/visualization.")
PY
