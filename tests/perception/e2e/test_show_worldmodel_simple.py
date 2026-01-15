#!/usr/bin/env python3
"""
Simple WorldModel Internal State Display

直接展示WorldModel的所有内部字段和实际值
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, '/media/yangyuhui/CODES1/Brain')

from brain.cognitive.world_model.world_model import WorldModel


def display_worldmodel_state(world_model: WorldModel):
    """显示WorldModel的完整内部状态"""
    print("\n" + "=" * 80)
    print("WorldModel 内部状态完整展示")
    print("=" * 80)
    print(f"展示时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 机器人状态
    print("\n" + "-" * 80)
    print("1. 🤖 机器人状态 (Robot State)")
    print("-" * 80)

    print(f"位置 (robot_position):")
    for key, value in world_model.robot_position.items():
        print(f"  {key}: {value}")

    print(f"\n速度 (robot_velocity):")
    for key, value in world_model.robot_velocity.items():
        print(f"  {key}: {value}")

    print(f"\n航向 (robot_heading): {world_model.robot_heading}°")
    print(f"电池 (battery_level): {world_model.battery_level}%")
    print(f"信号 (signal_strength): {world_model.signal_strength}%")

    # 2. 占据栅格
    print("\n" + "-" * 80)
    print("2. 🗺️ 占据栅格 (Occupancy Grid)")
    print("-" * 80)

    if world_model.current_map is not None:
        import numpy as np
        grid = world_model.current_map
        print(f"栅格形状: {grid.shape}")
        print(f"分辨率: {world_model.map_resolution} m/cell")
        print(f"原点: {world_model.map_origin}")

        total_cells = grid.size
        unknown_cells = np.sum(grid == -1)
        free_cells = np.sum(grid == 0)
        occupied_cells = np.sum(grid == 100)

        print(f"\n栅格统计:")
        print(f"  总单元数: {total_cells:,}")
        print(f"  未知 (-1): {unknown_cells:,} ({100*unknown_cells/total_cells:.1f}%)")
        print(f"  空闲 (0): {free_cells:,} ({100*free_cells/total_cells:.1f}%)")
        print(f"  占据 (100): {occupied_cells:,} ({100*occupied_cells/total_cells:.1f}%)")
    else:
        print("  (栅格未初始化)")

    # 3. 语义物体
    print("\n" + "-" * 80)
    print("3. 📦 语义物体 (Semantic Objects)")
    print("-" * 80)

    print(f"语义物体数量: {len(world_model.semantic_objects)}")
    print(f"最大容量: {world_model.max_semantic_objects}")

    if world_model.semantic_objects:
        print(f"\n物体列表:")
        for i, (obj_id, obj) in enumerate(list(world_model.semantic_objects.items())[:5]):
            print(f"\n  [{i+1}] ID: {obj_id}")
            print(f"      标签: {obj.label}")
            if hasattr(obj, 'world_position'):
                wx, wy = obj.world_position
                print(f"      世界位置: ({wx:.2f}, {wy:.2f})")
            print(f"      状态: {obj.state}")
            print(f"      置信度: {obj.confidence:.2f}")
            print(f"      描述: {obj.description[:50]}..." if len(obj.description) > 50 else f"      描述: {obj.description}")
            if hasattr(obj, 'first_seen'):
                print(f"      首次观测: {obj.first_seen.strftime('%H:%M:%S')}")
                print(f"      最后观测: {obj.last_seen.strftime('%H:%M:%S')}")
            print(f"      观测次数: {obj.observation_count}")
            print(f"      是否目标: {obj.is_target}")
    else:
        print("  (暂无语义物体)")

    # 4. 跟踪物体
    print("\n" + "-" * 80)
    print("4. 🎯 跟踪物体 (Tracked Objects)")
    print("-" * 80)

    print(f"跟踪物体数量: {len(world_model.tracked_objects)}")

    if world_model.tracked_objects:
        for obj_id, obj in list(world_model.tracked_objects.items())[:5]:
            print(f"  - {obj_id}: {obj}")
    else:
        print("  (暂无跟踪物体)")

    # 5. 探索前沿
    print("\n" + "-" * 80)
    print("5. 🔍 探索前沿 (Exploration Frontiers)")
    print("-" * 80)

    frontiers = world_model.exploration_frontiers
    print(f"前沿数量: {len(frontiers)}")
    print(f"最大前沿数: {world_model.max_frontiers}")

    if frontiers:
        print(f"\n前沿点:")
        for i, frontier in enumerate(frontiers[:5]):
            print(f"  [{i+1}] ID: {frontier.id}")
            print(f"      位置: ({frontier.center_x:.1f}, {frontier.center_y:.1f})")
            print(f"      优先级: {frontier.priority}")
            print(f"      单元格数: {frontier.size}")
    else:
        print("  (暂无前沿)")

    # 6. 位姿历史
    print("\n" + "-" * 80)
    print("6. 📍 位姿历史 (Pose History)")
    print("-" * 80)

    pose_history = world_model.pose_history
    print(f"历史记录数: {len(pose_history)}")
    print(f"最大历史数: {world_model.max_pose_history}")

    if pose_history:
        print(f"\n最近轨迹:")
        for i, pose_entry in enumerate(pose_history[-10:]):
            timestamp = pose_entry.get('timestamp', 'N/A')
            x = pose_entry.get('x', 0)
            y = pose_entry.get('y', 0)
            print(f"  [{i+1}] {timestamp}: ({x:.2f}, {y:.2f})")

    # 7. 环境信息
    print("\n" + "-" * 80)
    print("7. 🌤️ 环境信息 (Environment)")
    print("-" * 80)

    weather = world_model.weather
    print(f"天气: {weather.get('condition', 'unknown')}")
    print(f"风速: {weather.get('wind_speed', 0):.1f} m/s")
    print(f"风向: {weather.get('wind_direction', 0):.1f}°")
    print(f"能见度: {weather.get('visibility', 'unknown')}")
    print(f"温度: {weather.get('temperature', 0):.1f}°C")

    # 8. 变化历史
    print("\n" + "-" * 80)
    print("8. 📝 变化历史 (Change History)")
    print("-" * 80)

    change_history = world_model.change_history
    print(f"变化记录数: {len(change_history)}")

    if change_history:
        print(f"\n最近变化:")
        for change in change_history[-5:]:
            print(f"  - {change}")

    # 9. 元数据
    print("\n" + "-" * 80)
    print("9. ⚙️ 元数据 (Metadata)")
    print("-" * 80)

    print(f"对象计数器: {world_model._object_counter}")
    print(f"前沿计数器: {world_model._frontier_counter}")
    print(f"已探索位置数: {len(world_model.explored_positions)}")
    print(f"上次更新时间: {world_model.last_update}")

    print("\n" + "=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🎯 WorldModel 内部状态展示 (简化版)")
    print("=" * 80)

    # 初始化WorldModel
    world_config = {
        'map_resolution': 0.1,  # 10cm per cell
        'map_size': 50.0,      # 50m x 50m
    }

    print("\n初始化WorldModel...")
    world_model = WorldModel(config=world_config)

    print("✅ WorldModel初始化完成")
    print(f"   地图分辨率: {world_model.map_resolution}m/cell")
    print(f"   地图原点: {world_model.map_origin}")

    # 展示内部状态
    display_worldmodel_state(world_model)

    # 保存到JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"/media/yangyuhui/CODES1/Brain/tests/perception/e2e/worldmodel_state_simple_{timestamp}.json"

    # 收集数据
    data = {
        "metadata": {
            "capture_time": datetime.now().isoformat(),
        },
        "robot_state": {
            "position": world_model.robot_position,
            "velocity": world_model.robot_velocity,
            "heading": world_model.robot_heading,
            "battery": world_model.battery_level,
            "signal": world_model.signal_strength
        },
        "occupancy_grid": {
            "shape": world_model.current_map.shape if world_model.current_map is not None else None,
            "resolution": world_model.map_resolution,
            "origin": world_model.map_origin,
        },
        "semantic_objects": {
            "count": len(world_model.semantic_objects),
        },
        "exploration": {
            "frontiers_count": len(world_model.exploration_frontiers),
            "max_frontiers": world_model.max_frontiers,
            "explored_count": len(world_model.explored_positions)
        },
        "history": {
            "pose_history_count": len(world_model.pose_history),
            "change_history_count": len(world_model.change_history)
        },
        "environment": world_model.weather,
        "metadata_internal": {
            "object_counter": world_model._object_counter,
            "frontier_counter": world_model._frontier_counter,
            "last_update": str(world_model.last_update)
        }
    }

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n💾 数据已保存到: {json_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
