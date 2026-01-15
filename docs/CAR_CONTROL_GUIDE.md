# 小车控制使用指南

## 快速开始

### 1. 启动RViz2可视化

```bash
source /opt/ros/galactic/setup.bash
rviz2 -d config/rviz2/nova_carter_final.rviz
```

在RViz中你会看到：
- ✅ 小车坐标系（car3）
- ✅ 激光雷达点云
- ✅ RGB图像
- ✅ TF树结构

### 2. 控制小车移动

#### 方式1：直接命令（推荐用于测试）

```bash
# 前进5秒
source /opt/ros/galactic/setup.bash
timeout 5 ros2 topic pub /car3/twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.5}, angular: {z: 0.0}}" --rate 10

# 左转3秒
timeout 3 ros2 topic pub /car3/twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.5}}" --rate 10

# 停止
ros2 topic pub /car3/twist geometry_msgs/msg/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}" --once
```

#### 方式2：演示脚本

```bash
python3 scripts/control_car_demo.py
```

这将自动执行一系列动作：
1. 前进 3秒
2. 左转 2秒
3. 前进 3秒
4. 右转 2秒
5. 前进 2秒
6. 停止

#### 方式3：交互式控制（推荐）

```bash
python3 scripts/control_car_interactive.py
```

控制键：
- `W` / `↑` - 前进
- `S` / `↓` - 后退
- `A` / `←` - 左移（Y轴）
- `D` / `→` - 右移（Y轴）
- `Q` - 左转（原地）
- `E` - 右转（原地）
- `空格` - 停止
- `+` / `-` - 调整速度
- `ESC` - 退出

#### 方式4：里程计诊断（调试用）

```bash
python3 scripts/diagnose_odom_control.py
```

这将自动运行一系列测试：
1. X轴正向（前进）
2. X轴负向（后退）
3. Y轴正向（右移）
4. Y轴负向（左移）
5. 角速度（原地转向）
6. 组合运动（斜向）

每项测试都会显示期望速度 vs 实际里程计反馈，帮助诊断控制问题。

### 3. 查看话题数据

```bash
# 查看odometry
ros2 topic echo /car3/local_odom

# 查看激光雷达
ros2 topic echo /car3/lidar_points

# 查看图像
ros2 topic hz /car3/rgbImage
```

## 可用话题

| 话题 | 类型 | 描述 |
|------|------|------|
| `/car3/twist` | geometry_msgs/Twist | 控制指令（输入） |
| `/car3/local_odom` | nav_msgs/Odometry | 里程计 |
| `/car3/lidar_points` | sensor_msgs/PointCloud2 | 激光雷达点云 |
| `/car3/rgbImage` | sensor_msgs/Image | RGB图像 |

## Twist消息说明

```yaml
linear:
  x: 0.5    # 前进速度 (m/s)，正数前进，负数后退
  y: 0.0    # 侧向速度（通常为0）
  z: 0.0    # 垂直速度（通常为0）
angular:
  x: 0.0    # 翻滚（通常为0）
  y: 0.0    # 俯仰（通常为0）
  z: 0.5    # 偏航角速度 (rad/s)，正数左转，负数右转
```

## 常见问题

### Q: 小车不动？
A: 检查仿真节点是否运行：
```bash
ros2 node list | grep car3
```

### Q: Y轴（左右）移动不工作？
A: 使用里程计诊断工具检查：
```bash
python3 scripts/diagnose_odom_control.py
```
这会测试各个轴向并显示实际的里程计反馈。如果linear.y在里程计中始终为0，说明机器人配置不支持横向移动。

### Q: RViz中看不到小车？
A:
1. 检查Fixed Frame是否设置为`car3`或`map`
2. 添加TF显示
3. 添加Axes显示，Reference Frame设为`car3`

### Q: 如何调整速度？
A: 在交互式控制中按`+`/`-`，或修改消息中的`x`值：
- `x: 0.1` - 慢速
- `x: 0.5` - 中速
- `x: 1.0` - 快速

## 下一步

1. 将认知层集成到控制回路
2. 使用感知层输出进行避障
3. 实现自主导航
