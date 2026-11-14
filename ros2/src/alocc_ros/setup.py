import os
from glob import glob

from setuptools import setup

package_name = "alocc_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, package_name + ".connectors"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Person 3",
    maintainer_email="person3@todo.todo",
    description="ROS 2 integration framework for ALOcc occupancy and flow predictions.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "prediction_input_node = alocc_ros.prediction_input_node:main",
            "occupancy_publisher = alocc_ros.occupancy_publisher:main",
            "flow_publisher = alocc_ros.flow_publisher:main",
            "occupancy_subscriber = alocc_ros.occupancy_subscriber:main",
            "flow_subscriber = alocc_ros.flow_subscriber:main",
            "prediction_validator = alocc_ros.prediction_validator:main",
            "runtime_monitor = alocc_ros.runtime_monitor:main",
            "visualization_node = alocc_ros.visualization_node:main",
            "waymo_scene0_replay_node = alocc_ros.waymo_scene0_replay_node:main",
            "waymo_scene0_replay_check_node = alocc_ros.waymo_scene0_replay_check_node:main",
        ],
    },
)
