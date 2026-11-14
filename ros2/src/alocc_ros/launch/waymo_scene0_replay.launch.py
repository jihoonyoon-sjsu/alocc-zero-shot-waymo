from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _repo_root() -> Path:
    """Return the ros2 project directory that contains config/model.yaml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "model.yaml").is_file():
            return parent
    return here.parents[3]


def _text(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _bool(text: str) -> bool:
    return text.lower() == "true"


def _expand(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    return str(Path(text).expanduser())


def _nodes(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    repo = _repo_root()
    dataset_root = _expand(value("dataset_root"))
    dataset_path = _expand(value("dataset_path")) or dataset_root
    model_config = _expand(value("model_config")) or str(repo / "config" / "model.yaml")
    validation_csv = _expand(value("validation_csv")) or str(
        Path(dataset_path) / "reports" / "replay_validation.csv"
    )
    start_frame = int(value("start_frame"))
    end_frame = int(value("end_frame"))
    nodes = [
        Node(
            package="alocc_ros",
            executable="waymo_scene0_replay_check_node",
            name="waymo_scene0_replay_check_node",
            output="screen",
            parameters=[
                {
                    "expected_frames": end_frame - start_frame + 1,
                    "validation_csv": validation_csv,
                    "timeout_sec": float(value("timeout_sec")),
                }
            ],
        ),
        Node(
            package="alocc_ros",
            executable="waymo_scene0_replay_node",
            name="waymo_scene0_replay_node",
            output="screen",
            parameters=[
                {
                    "dataset_path": dataset_path,
                    "model_config": model_config,
                    "replay_rate": float(value("replay_rate")),
                    "loop": _bool(value("loop")),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "publish_camera_images": _bool(value("publish_camera_images")),
                    "publish_camera_info": _bool(value("publish_camera_info")),
                    "publish_occupancy": _bool(value("publish_occupancy")),
                }
            ],
        ),
    ]
    if _bool(value("enable_markers")):
        nodes.append(
            Node(
                package="alocc_ros",
                executable="visualization_node",
                name="waymo_scene0_visualization_node",
                output="screen",
                parameters=[
                    {
                        "mode": "pkl",
                        "model_config": model_config,
                        "ros_config": str(repo / "config" / "waymo_scene0_visualization.yaml"),
                        "results_dir": str(repo / "results" / "waymo_scene0_replay"),
                    }
                ],
            )
        )
    return nodes


def generate_launch_description():
    repo = _repo_root()
    defaults = yaml.safe_load((repo / "config" / "waymo_scene0_replay.yaml").read_text())
    dataset_root = _expand(str(defaults.get("dataset_root", "~/alocc-data/waymo_scene0")))
    names = (
        "replay_rate",
        "loop",
        "start_frame",
        "end_frame",
        "publish_camera_images",
        "publish_camera_info",
        "publish_occupancy",
        "timeout_sec",
        "enable_markers",
    )
    arguments = [
        DeclareLaunchArgument("dataset_root", default_value=dataset_root),
        # Empty values mean "derive from dataset_root and the repository".
        DeclareLaunchArgument("dataset_path", default_value=""),
        DeclareLaunchArgument("model_config", default_value=""),
        DeclareLaunchArgument("validation_csv", default_value=""),
    ]
    arguments.extend(
        DeclareLaunchArgument(name, default_value=_text(defaults[name])) for name in names
    )
    return LaunchDescription(arguments + [OpaqueFunction(function=_nodes)])
