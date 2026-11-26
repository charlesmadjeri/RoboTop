#!/usr/bin/env python3
"""so101.launch.py - the whole stack, one include per node launch file.

    ros2 launch so101_bringup so101.launch.py                               # simulation
    ros2 launch so101_bringup so101.launch.py use_sim:=false port:=<port>   # real arm
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def include(package: str, file: str, **arguments) -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare(package), "launch", file]),
        launch_arguments=arguments.items(),
    )


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim", default_value="true", description="Simulated backend or real robot"),
        DeclareLaunchArgument("port", default_value="", description="Serial port of the real SO-ARM101"),
        DeclareLaunchArgument("camera_index", default_value="0",
                              description="USB camera index of the real arm, -1 disables it"),
        DeclareLaunchArgument("camera_fov_deg", default_value="70.0",
                              description="Horizontal field of view used to estimate the camera intrinsics"),
        include("so101_driver", "driver.launch.py",
                use_sim=LaunchConfiguration("use_sim"), port=LaunchConfiguration("port"),
                camera_index=LaunchConfiguration("camera_index"),
                camera_fov_deg=LaunchConfiguration("camera_fov_deg")),
        include("so101_description", "viz.launch.py"),
        # Add your nodes here, e.g. include("so101_perception", "perception.launch.py"),
        include("so101_brain", "brain.launch.py"),
    ])
