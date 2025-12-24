# 占据栅格地图文档

Brain感知模块提供占据栅格地图生成功能，可以从多种传感器数据生成和维护环境地图。

## 占据栅格地图使用

### 基本使用

```python
from brain.perception.mapping.occupancy_mapper import OccupancyMapper

# 创建占据栅格地图
mapper = OccupancyMapper(
    resolution=0.1,      # 分辨率: 0.1米/栅格
    map_size=50.0,        # 地图大小: 50米x50米
    camera_fov=1.57,      # 相机视场角: 90度
    camera_range=10.0,     # 相机感知范围: 10米
    lidar_range=30.0,      # 激光雷达范围: 30米
    config={               # 更新参数
        "occupied_prob": 0.7,  # 占据概率阈值
        "free_prob": 0.3,     # 自由概率阈值
        "min_depth": 0.1,      # 最小深度(米)
        "max_depth": 10.0      # 最大深度(米)
    }
)

# 从深度图更新
depth_image = get_depth_image()  # (H, W) numpy数组
pose = (x, y, yaw)  # 机器人位姿 (x, y, yaw)
mapper.update_from_depth(depth_image, pose=pose)

# 从激光雷达更新
ranges, angles = get_laser_scan()  # 距离数组, 角度数组
mapper.update_from_laser(ranges, angles, pose=pose)

# 从点云更新
pointcloud = get_pointcloud()  # (N, 3) 或 (N, 6) numpy数组
mapper.update_from_pointcloud(pointcloud, pose=pose)
```

### 获取地图

```python
# 获取占据栅格地图
grid = mapper.get_grid()

# 访问地图数据
grid_data = grid.data  # (H, W) numpy数组, 值: -1=未知, 0=自由, 100=占据
resolution = grid.resolution  # 分辨率(米/栅格)
origin_x, origin_y = grid.origin_x, grid.origin_y  # 地图原点(米)

# 坐标转换
# 世界坐标 -> 栅格坐标
world_x, world_y = 5.0, 3.0  # 世界坐标(米)
grid_x, grid_y = grid.world_to_grid(world_x, world_y)

# 栅格坐标 -> 世界坐标
world_x, world_y = grid.grid_to_world(grid_x, grid_y)

# 检查栅格状态
if grid.is_occupied(grid_x, grid_y):
    print(f"位置({world_x}, {world_y})被占据")
elif grid.is_free(grid_x, grid_y):
    print(f"位置({world_x}, {world_y})可通行")
elif grid.is_unknown(grid_x, grid_y):
    print(f"位置({world_x}, {world_y})未知")

# 设置栅格状态
grid.set_cell(grid_x, grid_y, grid.CellState.OCCUPIED)  # 设置为占据
cell_state = grid.get_cell(grid_x, grid_y)  # 获取栅格状态
```

### 地图查询

```python
# 检查世界坐标位置是否被占据
if mapper.is_occupied_at(5.0, 3.0):
    print("世界坐标(5.0, 3.0)被占据")

if mapper.is_free_at(5.0, 3.0):
    print("世界坐标(5.0, 3.0)可通行")

# 获取最近障碍物
nearest = mapper.get_nearest_obstacle(
    x=0.0,       # 查询点x坐标
    y=0.0,       # 查询点y坐标
    max_range=5.0  # 最大搜索范围(米)
)

if nearest:
    obs_x, obs_y, distance = nearest
    print(f"最近障碍物: ({obs_x}, {obs_y}), 距离: {distance}米")
```

### 地图统计

```python
# 获取地图统计信息
stats = mapper.get_statistics()
print(f"总栅格数: {stats['total_cells']}")
print(f"占据栅格数: {stats['occupied']}")
print(f"自由栅格数: {stats['free']}")
print(f"未知栅格数: {stats['unknown']}")
print(f"地图覆盖率: {stats['coverage']*100:.1f}%")
print(f"分辨率: {stats['resolution']}米/栅格")
print(f"地图大小: {stats['map_size']}米")
```

## 地图配置

### 基本参数

```python
mapper = OccupancyMapper(
    resolution=0.1,      # 分辨率: 米/栅格
    map_size=50.0,        # 地图大小: 米
    camera_fov=1.57,      # 相机视场角: 弧度
    camera_range=10.0,     # 相机感知范围: 米
    lidar_range=30.0,      # 激光雷达范围: 米
)
```

### 相机参数

```python
# 相机内参(也可以通过配置提供)
mapper.camera_fx = 525.0  # 焦距x
mapper.camera_fy = 525.0  # 焦距y
mapper.camera_cx = 320.0  # 光心x
mapper.camera_cy = 240.0  # 光心y
```

### 更新参数

```python
mapper.occupied_prob = 0.7  # 占据概率阈值
mapper.free_prob = 0.3     # 自由概率阈值
mapper.min_depth = 0.1      # 最小深度(米)
mapper.max_depth = 10.0     # 最大深度(米)
```

## 数据来源

### 深度图

```python
# 从深度图更新占据栅格
depth_image = get_depth_image()  # (H, W) numpy数组
pose = (x, y, yaw)  # 机器人位姿

# 可选: 相机相对于机器人的位姿
camera_pose = (0.2, 0.0, 0.0)  # (x, y, yaw)

mapper.update_from_depth(
    depth_image=depth_image,
    pose=pose,
    camera_pose=camera_pose  # 可选
)
```

### 激光雷达

```python
# 从激光雷达更新占据栅格
ranges, angles = get_laser_scan()  # 距离数组, 角度数组
pose = (x, y, yaw)  # 机器人位姿

mapper.update_from_laser(
    ranges=ranges,
    angles=angles,
    pose=pose
)
```

### 点云

```python
# 从点云更新占据栅格
pointcloud = get_pointcloud()  # (N, 3) 或 (N, 6) numpy数组
pose = (x, y, yaw)  # 机器人位姿

mapper.update_from_pointcloud(
    pointcloud=pointcloud,
    pose=pose
)
```

## 地图可视化

```python
import matplotlib.pyplot as plt

# 获取地图数据
grid = mapper.get_grid()
grid_data = grid.data  # (H, W) numpy数组

# 可视化地图
plt.figure(figsize=(8, 8))
plt.imshow(grid_data, cmap='gray', origin='lower')
plt.title('占据栅格地图')
plt.colorbar(label='占据状态')
plt.show()

# 自定义颜色映射
import matplotlib.colors as mcolors
colors = ['white', 'lightgray', 'black']  # 未知, 自由, 占据
cmap = mcolors.ListedColormap(colors)
bounds = [-1, 0, 50, 100]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(8, 8))
plt.imshow(grid_data, cmap=cmap, norm=norm, origin='lower')
plt.title('占据栅格地图 (自定义颜色)')
plt.colorbar(label='占据状态')
plt.show()
```

## 地图操作

### 地图重置

```python
# 重置地图为全未知
mapper.reset()
```

### 地图保存与加载

```python
import pickle

# 保存地图
grid = mapper.get_grid()
with open('occupancy_grid.pkl', 'wb') as f:
    pickle.dump(grid, f)

# 加载地图
with open('occupancy_grid.pkl', 'rb') as f:
    loaded_grid = pickle.load(f)
    
# 使用加载的地图创建新映射器
new_mapper = OccupancyMapper()
new_mapper.grid = loaded_grid
```

### 地图融合

```python
# 融合两个地图
grid1 = mapper1.get_grid()
grid2 = mapper2.get_grid()

# 创建更大的合并地图
merged_width = max(grid1.width, grid2.width)
merged_height = max(grid1.height, grid2.height)
merged_grid = OccupancyGrid(merged_width, merged_height, grid1.resolution)

# 合并逻辑: 如果任一地图标记为占据，则合并地图为占据
for y in range(merged_height):
    for x in range(merged_width):
        if x < grid1.width and y < grid1.height:
            if grid1.data[y, x] == grid1.CellState.OCCUPIED:
                merged_grid.data[y, x] = grid1.CellState.OCCUPIED
                continue
                
        if x < grid2.width and y < grid2.height:
            if grid2.data[y, x] == grid2.CellState.OCCUPIED:
                merged_grid.data[y, x] = grid2.CellState.OCCUPIED
                continue
                
        # 否则保持未知
        merged_grid.data[y, x] = grid1.CellState.UNKNOWN
```

## 性能优化

### 降采样更新

```python
# 对深度图进行降采样，提高更新速度
import cv2

depth_image = get_depth_image()
# 降采样到一半分辨率
downsampled_depth = cv2.pyrDown(depth_image)

mapper.update_from_depth(downsampled_depth, pose)
```

### 限制更新范围

```python
# 只更新机器人周围一定范围内的地图
from brain.perception.utils.coordinates import transform_local_to_world

# 当前机器人位姿
robot_x, robot_y, robot_yaw = pose

# 感兴趣范围(米)
update_range = 20.0

# 只处理深度图中的有效区域
h, w = depth_image.shape
center_x, center_y = w // 2, h // 2
radius = int(update_range / mapper.resolution * 2)  # 近似像素半径

# 创建掩码
y, x = np.ogrid[:h, :w]
mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2

# 只更新掩码区域
if np.any(mask):
    # 这里需要修改OccupancyMapper实现以支持掩码
    pass
```

### 并行更新

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_update_map(mapper, depth_image, ranges, pointcloud, pose):
    with ThreadPoolExecutor() as executor:
        # 并行执行更新
        tasks = [
            asyncio.get_event_loop().run_in_executor(
                executor, mapper.update_from_depth, depth_image, pose
            ),
            asyncio.get_event_loop().run_in_executor(
                executor, mapper.update_from_laser, ranges, [], pose
            ),
            asyncio.get_event_loop().run_in_executor(
                executor, mapper.update_from_pointcloud, pointcloud, pose
            )
        ]
        await asyncio.gather(*tasks)

# 使用异步更新
await async_update_map(mapper, depth_image, ranges, pointcloud, pose)
```

## 故障处理

### 无效数据处理

```python
# 检查深度图有效性
if depth_image is None or depth_image.size == 0:
    print("深度图为空")
    return

# 检查激光数据有效性
if not ranges or not angles or len(ranges) != len(angles):
    print("激光数据无效")
    return

# 检查点云有效性
if pointcloud is None or pointcloud.size == 0:
    print("点云为空")
    return
```

### 坐标转换错误

```python
try:
    # 坐标转换
    world_x, world_y = transform_local_to_world(
        local_x, local_y, robot_x, robot_y, robot_yaw
    )
    gx, gy = grid.world_to_grid(world_x, world_y)
    
    # 检查栅格有效性
    if grid.is_valid(gx, gy):
        grid.set_cell(gx, gy, grid.CellState.OCCUPIED)
    else:
        print(f"栅格坐标超出范围: ({gx}, {gy})")
except Exception as e:
    print(f"坐标转换错误: {e}")
```

## 应用示例

### 动态地图更新

```python
async def dynamic_map_update(sensor_manager, mapper):
    while True:
        # 获取最新传感器数据
        perception_data = await sensor_manager.get_fused_perception()
        
        # 获取机器人位姿
        if perception_data.pose:
            pose = (perception_data.pose.x, perception_data.pose.y, perception_data.pose.yaw)
            
            # 从深度图更新
            if perception_data.depth_image is not None:
                mapper.update_from_depth(perception_data.depth_image, pose)
                
            # 从激光雷达更新
            if perception_data.laser_ranges and perception_data.laser_angles:
                mapper.update_from_laser(
                    perception_data.laser_ranges,
                    perception_data.laser_angles,
                    pose
                )
                
            # 从点云更新
            if perception_data.pointcloud is not None:
                mapper.update_from_pointcloud(perception_data.pointcloud, pose)
        
        # 控制更新频率
        await asyncio.sleep(0.1)  # 10Hz

# 启动动态地图更新
# asyncio.run(dynamic_map_update(sensor_manager, mapper))
```

## 相关文档

- [传感器接口](sensor_interfaces.md)
- [多传感器融合](sensor_fusion.md)
- [Isaac Sim集成](isaac_sim_integration.md)


