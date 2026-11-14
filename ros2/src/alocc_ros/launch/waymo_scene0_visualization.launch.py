"""Start the scene 0 replay, its marker node, and RViz.

The replay publishers come from waymo_scene0_replay.launch.py with that
launch's marker node left off, so this file does not start a second copy.
Markers use config/waymo_scene0_rviz.yaml unless full_resolution is true.
RViz starts only when DISPLAY is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    Shutdown,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _repo_root() -> Path:
    """Return the ros2 project directory that contains config/model.yaml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "model.yaml").is_file():
            return parent
    return here.parents[3]


def _bool(text: str) -> bool:
    return text.lower() == "true"


def _actions(context):
    repo = _repo_root()
    share = Path(get_package_share_directory("alocc_ros"))
    replay = share / "launch" / "waymo_scene0_replay.launch.py"
    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(replay)),
            launch_arguments={
                "enable_markers": "false",
                "dataset_root": LaunchConfiguration("dataset_root").perform(context),
                "dataset_path": LaunchConfiguration("dataset_path").perform(context),
                "model_config": LaunchConfiguration("model_config").perform(context),
                "loop": LaunchConfiguration("loop").perform(context),
                "start_frame": LaunchConfiguration("start_frame").perform(context),
                "end_frame": LaunchConfiguration("end_frame").perform(context),
                "replay_rate": LaunchConfiguration("replay_rate").perform(context),
                "validation_csv": LaunchConfiguration("validation_csv").perform(context),
            }.items(),
        )
    ]
    if _bool(LaunchConfiguration("enable_markers").perform(context)):
        full_resolution = _bool(LaunchConfiguration("full_resolution").perform(context))
        config_name = (
            "waymo_scene0_visualization.yaml" if full_resolution else "waymo_scene0_rviz.yaml"
        )
        actions.append(
            Node(
                package="alocc_ros",
                executable="visualization_node",
                name="waymo_scene0_visualization_node",
                output="screen",
                parameters=[
                    {
                        "mode": "pkl",
                        "model_config": str(repo / "config" / "model.yaml"),
                        "ros_config": str(repo / "config" / config_name),
                        "results_dir": str(repo / "results" / "waymo_scene0_rviz"),
                    }
                ],
            )
        )
    if _bool(LaunchConfiguration("start_rviz").perform(context)):
        display = os.environ.get("DISPLAY", "").strip()
        if display:
            rviz = Node(
                package="rviz2",
                executable="rviz2",
                name="waymo_scene0_rviz",
                output="screen",
                arguments=["-d", str(repo / "visualization" / "waymo_scene0.rviz")],
            )
            actions.append(rviz)
            actions.append(
                RegisterEventHandler(
                    OnProcessExit(target_action=rviz, on_exit=[Shutdown(reason="RViz closed")])
                )
            )
        else:
            actions.append(
                LogInfo(
                    msg=(
                        "DISPLAY is unset, so RViz was not started. "
                        "The replay and marker nodes can still run. "
                        "Open an X11-forwarded session and run this launch again to show the window."
                    )
                )
            )
    return actions


def generate_launch_description():
    repo = _repo_root()
    defaults = yaml.safe_load((repo / "config" / "waymo_scene0_replay.yaml").read_text())
    dataset_root = str(Path(str(defaults.get("dataset_root", "~/alocc-data/waymo_scene0"))).expanduser())
    received = repo / "results" / "waymo_scene0_rviz" / "received.csv"
    return LaunchDescription(
        [
            DeclareLaunchArgument("dataset_root", default_value=dataset_root),
            DeclareLaunchArgument("dataset_path", default_value=""),
            DeclareLaunchArgument("model_config", default_value=""),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("start_frame", default_value="0"),
            DeclareLaunchArgument("end_frame", default_value="39"),
            DeclareLaunchArgument("replay_rate", default_value="1.0"),
            DeclareLaunchArgument("enable_markers", default_value="true"),
            DeclareLaunchArgument("full_resolution", default_value="false"),
            DeclareLaunchArgument("start_rviz", default_value="true"),
            DeclareLaunchArgument("validation_csv", default_value=str(received)),
            OpaqueFunction(function=_actions),
        ]
    )
