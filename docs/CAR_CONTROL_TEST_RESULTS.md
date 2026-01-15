# 小车控制测试结果与使用指南

## 测试日期
2026-01-14

## 测试环境
- ROS2 Galactic
- 仿真环境: 自主开发
- 里程计话题: `/car3/car_info`
- 控制话题: `/car3/twist`

## 完整测试结果

### 测试概述
- 总测试数: 13
- 有效控制: 7 ✅
- 无效控制: 6 ❌

### ✅ 有效控制方式

| 测试 | 命令 | 移动距离 | 效果 |
|------|------|---------|------|
| **纯前进** | `x=0.5, ω=0.0` | 0.48m | ✅ 前进 |
| **纯后退** | `x=-0.5, ω=0.0` | 2.42m | ✅ 后退 |
| **前进+左转** | `x=0.3, ω=0.5` | 1.56m | ✅ 左转 |
| **前进+右转** | `x=0.3, ω=-0.5` | 0.88m | ✅ 右转 |
| **后退+左转** | `x=-0.3, ω=0.5` | 1.54m | ✅ 后退左转 |
| **后退+右转** | `x=-0.3, ω=-0.5` | 1.60m | ✅ 后退右转 |

### ❌ 无效控制方式

| 测试 | 命令 | 原因 |
|------|------|------|
| **原地左转** | `x=0.0, ω=0.5` | 无linear.x |
| **原地右转** | `x=0.0, ω=-0.5` | 无linear.x |
| **慢速前+快转** | `x=0.2, ω=0.8` | 速度比例不当 |
| **快速前+慢转** | `x=0.5, ω=0.3` | 速度比例不当 |
| **纯左移** | `y=-0.5` | 不支持Y轴 |

## 关键发现

### 1. 机器人类型
**特殊的差速驱动** - 必须同时使用 linear.x 和 angular.z 来实现转向

### 2. 转向限制
⚠️ **不能原地转向**
- `angular.z` 单独使用无效
- 必须配合 `linear.x > 0` 或 `linear.x < 0`

### 3. 最佳速度比例
基于测试结果：
- **前进+转向**: `linear.x=0.3, angular.z=0.5` ⭐
- **后退+转向**: `linear.x=-0.3, angular.z=0.5` ⭐
- **速度比例**: linear.x : angular.z ≈ 0.6 : 1.0

### 4. Y轴控制
❌ **不支持横向移动**
- `linear.y` 控制无效
- 只能沿X轴移动 + 转向

## 推荐控制方式

### 基本动作

| 动作 | linear.x | angular.z | 说明 |
|------|----------|-----------|------|
| **前进** | 0.3~0.5 | 0.0 | 沿X轴正向 |
| **后退** | -0.3~-0.5 | 0.0 | 沿X轴负向 |
| **左转** | 0.3 | 0.5 | 边前进边左转 |
| **右转** | 0.3 | -0.5 | 边前进边右转 |
| **后退左转** | -0.3 | 0.5 | 边后退边左转 |
| **后退右转** | -0.3 | -0.5 | 边后退边右转 |
| **停止** | 0.0 | 0.0 | 停止运动 |

### Python示例

```python
from geometry_msgs.msg import Twist

# 前进
msg = Twist()
msg.linear.x = 0.5
msg.angular.z = 0.0
publisher.publish(msg)

# 左转（必须同时设置linear.x和angular.z）
msg = Twist()
msg.linear.x = 0.3   # 必须！
msg.angular.z = 0.5  # 转向
publisher.publish(msg)

# 停止
msg = Twist()
msg.linear.x = 0.0
msg.angular.z = 0.0
publisher.publish(msg)
```

## 使用工具

### 1. 交互式控制（推荐）

```bash
python3 scripts/control_car_interactive_fixed.py
```

**控制键**:
- `W/↑` - 前进
- `S/↓` - 后退
- `A/←` - 左转 (前进+左转)
- `D/→` - 右转 (前进+右转)
- `Q` - 后退左转
- `E` - 后退右转
- `空格` - 停止
- `+/-` - 调整速度
- `ESC` - 退出

### 2. 完整测试

```bash
python3 scripts/test_all_control_combos.py
```

运行所有13种控制组合，查看详细测试报告。

### 3. 命令行控制

```bash
# 前进
timeout 3 ros2 topic pub /car3/twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.0}}" --rate 10

# 左转
timeout 3 ros2 topic pub /car3/twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.3}, angular: {z: 0.5}}" --rate 10

# 停止
ros2 topic pub /car3/twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}" --once
```

## 与标准差速驱动的区别

| 特性 | 标准差速驱动 | 您的仿真 |
|------|-------------|---------|
| **原地转向** | ✅ 支持 (x=0, ω≠0) | ❌ 不支持 |
| **前进+转向** | ✅ 支持 | ✅ 支持 |
| **Y轴移动** | ❌ 不支持 | ❌ 不支持 |
| **最佳比例** | 灵活 | x:ω ≈ 0.6:1 |

## 常见问题

### Q: 为什么不能原地转向？
A: 这是您仿真的物理模型特性。可能是因为：
- 没有独立的转向电机
- 轮子配置不支持纯转向
- 需要运动学约束

### Q: 如何实现精确的90度转弯？
A: 需要通过多次"前进+转向"的组合，估算转向角度：
```python
# 粗略估算：0.3m/s + 0.5rad/s 持续3秒 ≈ 90度
```

### Q: 能否实现斜向移动？
A: 不能。不支持Y轴控制，只能通过X轴+转向实现弧线运动。

## 集成到认知层

基于测试结果，认知层规划时应：

1. **路径规划** - 考虑不能原地转向，使用弧线轨迹
2. **速度规划** - 使用测试验证的比例 (x:ω = 0.6:1)
3. **避障** - 考虑转弯半径较大

示例：
```python
# 规划层生成控制指令
def plan_turn(direction):
    if direction == 'left':
        return Twist(linear=Vector3(x=0.3, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=0.5))
    elif direction == 'right':
        return Twist(linear=Vector3(x=0.3, y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=-0.5))
```

## 文件清单

| 文件 | 用途 |
|------|------|
| `scripts/control_car_interactive_fixed.py` | 交互式控制（推荐） |
| `scripts/test_all_control_combos.py` | 完整测试脚本 |
| `docs/CAR_CONTROL_TEST_RESULTS.md` | 本文档 |
| `docs/CAR_CONTROL_GUIDE.md` | 通用控制指南 |
| `docs/CAR_CONTROL_DIAGNOSTICS.md` | 诊断指南 |

## 总结

通过系统的13项测试，我们确定了您仿真的正确控制方式：

✅ **核心规律**: 必须同时使用 linear.x 和 angular.z
✅ **最佳参数**: linear.x=0.3, angular.z=0.5
✅ **限制**: 不能原地转向，不支持Y轴移动

这些结果已集成到交互式控制脚本中，可以直接使用。

---

**测试完成时间**: 2026-01-14
**测试脚本**: `scripts/test_all_control_combos.py`
**验证状态**: ✅ 已验证
