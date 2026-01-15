#!/usr/bin/env python3
"""
SLAM集成启动文件

启动SLAM节点和认知层订阅节点
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    生成SLAM集成启动描述

    启动顺序：
    1. FAST-LIVO SLAM节点
    2. 认知层节点（订阅SLAM输出）
    """

    # 声明启动参数
    slam_backend_arg = DeclareLaunchArgument(
        'slam_backend',
        default_value='fast_livo',
        description='SLAM后端选择: fast_livo, lio_sam, cartographer'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(
            os.path.dirname(__file__),
            '../config/slam_config.yaml'
        ),
        description='SLAM配置文件路径'
    )

    # FAST-LIVO SLAM节点
    # 注意：这里假设FAST-LIVO已经作为ROS2包安装
    # 如果没有安装，需要先编译FAST-LIVO源码
    fast_livo_node = Node(
        package='fast_livo',
        executable='fast_livo_node',
        name='fast_livo',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'lidar_topic': '/velodyne_points',
                'image_topic': '/camera/rgb/image_raw',
                'imu_topic': '/imu/data',
                'map_update_interval': 0.1,
            }
        ],
        remappings=[
            ('/points', '/velodyne_points'),
            ('/image_raw', '/camera/rgb/image_raw'),
            ('/imu', '/imu/data'),
        ]
    )

    # 备选：如果FAST-LIVO不可用，使用slam_toolbox
    slam_toolbox_node = Node(
        parameters=[
          os.path.join(get_package_share_directory('slam_toolbox'), 'config', 'mapper_params_online_async.yaml'),
          {'use_sim_time': False}
        ],
        package='slam_toolbox',
        executable='online_async_node',
        name='slam_toolbox',
        output='screen'
    )

    # Brain认知层节点
    # 这个节点订阅SLAM发布的地图和位姿
    brain_cognitive_node = Node(
        package='brain',
        executable='cognitive_layer_node',  # 需要在Brain的setup.py中定义
        name='brain_cognitive_layer',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'slam_backend': LaunchConfiguration('slam_backend'),
                'zero_copy': True,
                'update_frequency': 10.0,
            }
        ],
        remappings=[
            ('/slam/map', '/map'),
            ('/slam/pose', '/pose'),
            ('/slam/path', '/path'),
        ]
    )

    # RViz2可视化
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', os.path.join(
            os.path.dirname(__file__),
            '../../config/rviz2/slam_integration.rviz'
        )]
    )

    # TF2静态变换发布器
    # 发布传感器之间的静态变换
    static_transform_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_sensor_transform',
        arguments=[
            '0', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw
            'base_link',
            'sensor_link'
        ],
        parameters={'use_sim_time': False}
    )

    # 启动描述
    ld = LaunchDescription()

    # 添加参数声明
    ld.add_action(slam_backend_arg)
    ld.add_action(config_file_arg)

    # 添加TF2静态变换
    ld.add_action(static_transform_publisher)

    # 根据SLAM后端选择启动相应节点
    # 注意：这里使用条件执行需要launch_ros的扩展功能
    # 简化起见，注释掉FAST-LIVO节点，需要时手动取消注释

    # ld.add_action(fast_livo_node)  # 取消注释以启用FAST-LIVO
    # ld.add_action(slam_toolbox_node)  # 或使用slam_toolbox作为备选

    # 添加认知层节点
    # ld.add_action(brain_cognitive_node)  # 需要先实现cognitive_layer_node

    # 添加RViz
    # ld.add_action(rviz_node)  # 可选

    return ld


if __name__ == '__main__':
    generate_launch_description()
