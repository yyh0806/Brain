#!/usr/bin/env python3
"""
ROS2启动文件 - Brain节点

用于在ROS2环境中启动Brain感知驱动导航系统

使用方法:
    ros2 launch brain brain_node.launch.py
    
    # 指定配置文件
    ros2 launch brain brain_node.launch.py config:=/path/to/config.yaml
    
    # 指定目标
    ros2 launch brain brain_node.launch.py target:="前面的建筑门口"
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """生成启动描述"""
    
    # 声明参数
    config_arg = DeclareLaunchArgument(
        'config',
        default_value='',
        description='配置文件路径'
    )
    
    target_arg = DeclareLaunchArgument(
        'target',
        default_value='前面建筑的门口',
        description='导航目标描述'
    )
    
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='real',
        description='运行模式: real 或 simulation'
    )
    
    # Brain节点（通过Python脚本启动）
    brain_node = ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'examples',
                'ros2_navigation_demo.py'
            ),
            '--mode', LaunchConfiguration('mode'),
            '--target', LaunchConfiguration('target'),
            '--demo', 'navigation'
        ],
        output='screen'
    )
    
    return LaunchDescription([
        config_arg,
        target_arg,
        mode_arg,
        brain_node
    ])

