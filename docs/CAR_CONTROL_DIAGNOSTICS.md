# 小车控制诊断指南

## 诊断工具使用说明

### 快速启动

```bash
# 确保ROS2环境已加载
source /opt/ros/galactic/setup.bash

# 运行诊断脚本
python3 scripts/diagnose_odom_control.py
```

### 诊断流程

诊断脚本会自动执行以下步骤：

#### 步骤1：检查机器人配置
- 列出所有ROS2节点
- 检查car3相关话题
- 验证参数服务器配置

#### 步骤2：等待里程计数据
- 订阅 `/car3/local_odom` 话题
- 显示初始位置和速度
- 验证数据接收正常

#### 步骤3：控制测试序列
自动运行6项测试：

| 测试 | 命令 | 预期结果 | 验证内容 |
|------|------|----------|----------|
| **X轴正向** | linear.x=0.3 | 前进 | 前进功能 |
| **X轴负向** | linear.x=-0.3 | 后退 | 后退功能 |
| **Y轴正向** | linear.y=0.3 | 右移 | **横向移动（关键）** |
| **Y轴负向** | linear.y=-0.3 | 左移 | **横向移动（关键）** |
| **角速度** | angular.z=0.5 | 原地左转 | 转向功能 |
| **组合运动** | linear.x=0.2, linear.y=0.2 | 斜向移动 | 多轴协同 |

### 结果分析

每项测试会显示：

```
【测试】Y轴正向 - 右移（关键测试）
命令: linear.x=0.00, linear.y=0.30, angular.z=0.00
持续时间: 2.0秒
----------------------------------------------------------------------
初始位置: x=0.1234, y=0.5678

结果分析:
  位置变化: Δx=0.0012m, Δy=0.5834m
  平均速度反馈:
    linear.x = 0.0012 m/s (期望: 0.00)
    linear.y = 0.2891 m/s (期望: 0.30)  ✓
    angular.z = 0.0001 rad/s (期望: 0.00)
  ✓ Y轴控制正常
  ✓✓✓ 测试通过
```

### 故障诊断

#### 情况1：所有轴都失败

**症状**：所有测试的里程计反馈都接近0

**可能原因**：
- 仿真节点未运行
- 控制器未启动
- `/car3/twist` 话题未连接

**解决方法**：
```bash
# 检查节点
ros2 node list | grep car3

# 检查话题连接
ros2 topic info /car3/twist -v

# 检查是否有数据
ros2 topic hz /car3/local_odom
```

#### 情况2：X轴正常，Y轴失败

**症状**：
- ✓ 前进/后退测试通过
- ✗ 左/右移动测试失败（linear.y反馈≈0）

**可能原因**：
1. **机器人不是全向轮**
   - 麦克纳姆轮：支持全向移动
   - 普通轮：只能前进+转向

2. **控制器配置问题**
   - 控制器未启用linear.y
   - diff_drive控制器不支持横向移动

3. **URDF配置错误**
   - 轮型配置不正确
   - 传动比设置错误

**验证方法**：
```bash
# 检查控制器类型
ros2 param get /controller_name type

# 查看机器人描述
ros2 topic echo /robot_description | grep -i type
```

**解决方法**：
- 如果是普通轮机器人：使用 `control_car_demo.py`（差速驱动版本）
- 如果是麦克纳姆轮：检查控制器配置，应该使用 `omni_drive` 控制器

#### 情况3：角速度失败

**症状**：
- ✓ linear.x控制正常
- ✗ angular.z控制失败

**可能原因**：
- 轮子电机未连接
- 转向控制器未启动

#### 情况4：速度比例不对

**症状**：控制响应但速度比例失调

**可能原因**：
- 控制器增益参数错误
- 物理参数（轮径、轴距）配置错误

**调试方法**：
```bash
# 检查控制器参数
ros2 param list /controller_name

# 查看特定参数
ros2 param get /controller_name linear_scale
ros2 param get /controller_name angular_scale
```

### 参数服务器检查

诊断脚本会自动检查以下内容：

1. **节点列表**：确认仿真节点运行
2. **话题列表**：确认必要话题存在
3. **里程计数据**：实时反馈

手动检查参数：
```bash
# 列出所有参数
ros2 param list

# 查看特定节点参数
ros2 param list /car3_controller

# 获取参数值
ros2 param get /car3_controller wheel_radius
ros2 param get /car3_controller wheel_separation
```

### 常见机器人类型

| 类型 | linear.x | linear.y | angular.z | 控制器 |
|------|----------|----------|-----------|--------|
| **差速驱动** | ✓ | ✗ | ✓ | diff_drive |
| **全向轮** | ✓ | ✓ | ✓ | omni_drive |
| **阿克曼** | ✓ | ✗ | ✓ | ackermann_drive |

### RViz可视化辅助

在运行诊断的同时，打开RViz观察：

```bash
rviz2 -d config/rviz2/nova_carter_final.rviz
```

在RViz中：
- 观察 `car3` 坐标系移动
- 查看 `Odometry` 显示
- 检查 `TF` 树

### 下一步

根据诊断结果：

1. **如果是配置问题**：修改控制器配置文件
2. **如果是机器人类型限制**：使用适合的控制方式
3. **如果是控制器bug**：考虑更换控制器或上报问题
4. **如果一切正常**：开始集成认知层控制

### 实时监控

单独监控里程计数据：

```bash
ros2 topic echo /car3/local_odom --field pose.pose.position,header.stamp
```

监控控制命令：

```bash
ros2 topic echo /car3/twist
```

### 联系与支持

如果遇到未在此文档中描述的问题：

1. 检查 `brain/cognitive/` 下的认知层日志
2. 查看ROS2日志：`ros2 log list`
3. 运行完整系统测试：`pytest tests/integration/`

---

**最后更新**: 2026-01-14
**版本**: 1.0
