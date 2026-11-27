#!/usr/bin/env python3
"""perception.launch.py - launches the perception node

    ros2 launch so101_perception perception.launch.py                                   # camera pose from the TF
    ros2 launch so101_perception perception.launch.py camera_pose_file:=camera_pose.json # calibrated camera
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("camera_pose_file", default_value="",
                              description="Camera pose file of calibrate_camera, empty uses the TF of the driver"),
        Node(
            package="so101_perception",
            executable="perception",
            name="perception_node",
            output="screen",
            parameters=[{"camera_pose_file": LaunchConfiguration("camera_pose_file")}],
        ),
    ])
