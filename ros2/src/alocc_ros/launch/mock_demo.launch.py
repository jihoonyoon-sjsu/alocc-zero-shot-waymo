from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    parameters = [
        {
            "mode": LaunchConfiguration("mode").perform(context),
            "model_config": LaunchConfiguration("model_config").perform(context),
            "ros_config": LaunchConfiguration("ros_config").perform(context),
            "results_dir": LaunchConfiguration("results_dir").perform(context),
        }
    ]
    names = [
        "prediction_input_node",
        "occupancy_subscriber",
        "flow_subscriber",
        "prediction_validator",
        "runtime_monitor",
        "visualization_node",
    ]
    return [
        Node(
            package="alocc_ros",
            executable=name,
            name=name,
            output="screen",
            parameters=parameters,
        )
        for name in names
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="mock"),
            DeclareLaunchArgument("model_config", default_value=""),
            DeclareLaunchArgument("ros_config", default_value=""),
            DeclareLaunchArgument("results_dir", default_value="results"),
            OpaqueFunction(function=_nodes),
        ]
    )
