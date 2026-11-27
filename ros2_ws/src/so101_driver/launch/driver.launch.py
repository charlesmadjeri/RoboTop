#!/usr/bin/env python3
"""driver.launch.py - launches the SO-ARM101 driver node

    ros2 launch so101_driver driver.launch.py                # simulation
    ros2 launch so101_driver driver.launch.py use_sim:=false port:=<port> # real robot
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim", default_value="true",
                              description="Simulated backend or real robot"),
        DeclareLaunchArgument("port", default_value="",
                              description="Serial port of the real SO-ARM101 (lerobot-find-port)"),
        DeclareLaunchArgument("camera_index", default_value="0",
                              description="USB camera index of the real arm, -1 disables it"),
        DeclareLaunchArgument("camera_fov_deg", default_value="70.0",
                              description="Horizontal field of view used to estimate the camera intrinsics"),
        DeclareLaunchArgument("camera_width", default_value="640", description="Camera image width"),
        DeclareLaunchArgument("camera_height", default_value="480", description="Camera image height"),
        DeclareLaunchArgument("camera_fourcc", default_value="",
                              description="Camera pixel format, such as MJPG, empty is automatic"),
        Node(
            package="so101_driver",
            executable="driver",
            name="so101_driver",
            output="screen",
            parameters=[{
                "use_sim": ParameterValue(LaunchConfiguration("use_sim"), value_type=bool),
                "port": LaunchConfiguration("port"),
                "camera_index": ParameterValue(LaunchConfiguration("camera_index"), value_type=int),
                "camera_fov_deg": ParameterValue(LaunchConfiguration("camera_fov_deg"), value_type=float),
                "camera_width": ParameterValue(LaunchConfiguration("camera_width"), value_type=int),
                "camera_height": ParameterValue(LaunchConfiguration("camera_height"), value_type=int),
                "camera_fourcc": ParameterValue(LaunchConfiguration("camera_fourcc"), value_type=str),
            }],
        ),
    ])
