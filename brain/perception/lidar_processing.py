"""
点云处理模块 - Point Cloud Processing

负责处理 Isaac Sim Nova Carter 的 3D 激光雷达点云数据
功能：
- 点云数据解析（从 ROS2 PointCloud2 消息）
- 点云过滤（地面分离、噪声去除）
- 点云降采样（减少数据量）
- 点云到占据地图转换
- 点云保存和加载
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


@dataclass
class PointCloudData:
    """点云数据"""
    points: np.ndarray  # (N, 3) 或 (N, 4) numpy 数组
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_points": self.points.shape[0] if self.points is not None else 0,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "shape": self.points.shape if self.points is not None else None
        }


class LidarProcessor:
    """
    激光雷达点云处理器
    
    提供点云处理的各种功能
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # 配置参数
        self.max_points = self.config.get("max_points", 100000)
        self.downsample_factor = self.config.get("downsample_factor", 2)
        self.ground_z_threshold = self.config.get("ground_z_threshold", 0.05)  # 地面高度阈值（米）
        self.noise_stddev_threshold = self.config.get("noise_stddev", 0.01)  # 噪声标准差阈值
        
        # 统计信息
        self.stats = {
            "processed_count": 0,
            "total_points": 0,
            "filtered_points": 0
        }
        
        logger.info("LidarProcessor 初始化完成")
    
    def parse_pointcloud2(self, msg) -> np.ndarray:
        """
        解析 ROS2 PointCloud2 消息
        
        Args:
            msg: sensor_msgs/PointCloud2 消息
            
        Returns:
            np.ndarray: 点云数组 (N, 3)
        """
        try:
            # 解析点云数据
            data = np.frombuffer(msg.data, dtype=np.float32)
            
            # 获取点云维度和字段信息
            point_step = msg.point_step
            fields = msg.fields
            
            # 查找 x, y, z 字段的偏移
            x_offset = 0
            y_offset = 0
            z_offset = 0
            
            for field in fields:
                if field.name == "x":
                    x_offset = field.offset
                elif field.name == "y":
                    y_offset = field.offset
                elif field.name == "z":
                    z_offset = field.offset
            
            # 根据是否有强度或颜色信息调整
            # PointCloud2 可以有 x, y, z, intensity, rgb 等字段
            # 这里我们只提取 xyz 坐标
            
            # 计算点数
            num_points = len(data) // (3 * 4)  # 假设 xyz 是 float32 (4 bytes)
            
            # 重构点云
            if x_offset != 0 or y_offset != 0 or z_offset != 0:
                # 有偏移，需要特殊处理
                points = np.zeros((num_points, 3), dtype=np.float32)
                points[:, 0] = np.frombuffer(data[x_offset::point_step], dtype=np.float32)
                points[:, 1] = np.frombuffer(data[y_offset::point_step], dtype=np.float32)
                points[:, 2] = np.frombuffer(data[z_offset::point_step], dtype=np.float32)
            else:
                # xyz 连续存储
                points = np.frombuffer(data, dtype=np.float32)
                points = points.reshape(-1, 3)[:num_points, :]
            
            logger.debug(f"解析点云: {num_points} 个点, shape: {points.shape}")
            return points
            
        except Exception as e:
            logger.error(f"点云解析失败: {e}")
            return np.zeros((0, 3), dtype=np.float32)
    
    def remove_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        移除地面点
        
        使用统计方法：假设地面点主要在最低高度
        
        Args:
            points: 点云数组 (N, 3)
            
        Returns:
            ground, nonground: 分离的地面点和非地面点
        """
        if len(points) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        
        # 计算 z 坐标的统计信息
        z_values = points[:, 2]
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        z_min = np.min(z_values)
        
        # 地面点阈值：低于平均值减去一些标准差
        ground_threshold = z_mean - 0.5 * z_std
        
        # 分离地面和非地面点
        ground_mask = points[:, 2] < ground_threshold
        nonground_mask = ~ground_mask
        
        ground_points = points[ground_mask]
        nonground_points = points[nonground_mask]
        
        logger.info(f"地面分离: 地面 {len(ground_points)} 点, 非地面 {len(nonground_points)} 点")
        
        return ground_points, nonground_points
    
    def remove_noise(self, points: np.ndarray) -> np.ndarray:
        """
        移除噪声点（孤立点）
        
        使用统计方法：距离每个点最近的 k 个邻居的平均距离
        
        Args:
            points: 点云数组 (N, 3)
            
        Returns:
            filtered: 过滤后的点云
        """
        if len(points) < 10:
            return points
        
        # 使用简化的噪声检测
        # 计算每个点到其最近邻域的距离
        # 对于小点云，使用 k=5
        
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto')
            nbrs.fit(points)
            distances, _ = nbrs.kneighbors(points)
            
            # 平均距离
            mean_distance = np.mean(distances[:, 1:])
            
            # 标准差
            std_distance = np.std(distances[:, 1:])
            
            # 噪声阈值：距离均值超过 2 倍标准差
            noise_threshold = mean_distance + 2.0 * std_distance
            
            noise_mask = distances[:, 0] > noise_threshold
            filtered_points = points[~noise_mask]
            
            logger.info(f"噪声过滤: 移除 {np.sum(noise_mask)} 个噪声点")
            
            return filtered_points
            
        except ImportError:
            logger.warning("scikit-learn 不可用，跳过噪声过滤")
            return points
    
    def downsample(self, points: np.ndarray, factor: int = 2) -> np.ndarray:
        """
        降采样点云
        
        使用体素栅格降采样减少点数
        
        Args:
            points: 点云数组 (N, 3)
            factor: 降采样因子
            
        Returns:
            downsampled: 降采样后的点云
        """
        if len(points) == 0 or factor <= 1:
            return points
        
        try:
            from sklearn.neighbors import RadiusNeighbors
            from scipy.spatial import KDTree
        except ImportError:
            logger.warning("scipy/sklearn 不可用，使用简单的降采样")
            # 简单降采样：每隔 n 个点取一个
            downsampled = points[::factor]
            logger.info(f"简单降采样: {len(points)} -> {len(downsampled)} (factor={factor})")
            return downsampled
        
        # 使用 KDTree 进行降采样
        tree = KDTree(points)
        
        # 体素大小：根据降采样因子估算
        # 假设原始点云的平均密度，计算合适的体素大小
        # 这里使用简单方法：保留最近的点
        downsampled_indices = []
        indices_to_keep = set()
        
        for i, point in enumerate(points):
            if i in indices_to_keep:
                continue
            
            # 查询最近的邻居
            dist, idx = tree.query(point, k=factor)
            
            # 只保留自己（避免重复）和足够远的点
            # 实际降采样会保留点云的主要结构
        
        # 简化：每隔 factor 个点取一个
        downsampled = points[::factor]
        
        logger.info(f"降采样: {len(points)} -> {len(downsampled)} (factor={factor})")
        return downsampled
    
    def to_occupancy_grid(
        self, 
        points: np.ndarray,
        origin: Tuple[float, float] = (0.0, 0.0),
        resolution: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        将点云转换为占据网格地图
        
        Args:
            points: 点云数组 (N, 3)
            origin: 地图原点 (x, y)
            resolution: 栅格分辨率 (米/格)
            
        Returns:
            grid: 占据网格 (H x W)
            info: 地图元信息
        """
        if len(points) == 0:
            return np.zeros((1, 1)), {"resolution": resolution, "width": 1, "height": 1}
        
        # 创建占据网格
        # 根据点云范围计算网格大小
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
        
        # 扩展范围（包括一些边界）
        x_range = x_max - x_min + 10.0
        y_range = y_max - y_min + 10.0
        
        # 转换为相对于原点的坐标
        x_rel = points[:, 0] - origin[0]
        y_rel = points[:, 1] - origin[1]
        
        # 计算网格尺寸
        width = int(x_range / resolution)
        height = int(y_range / resolution)
        
        # 初始化占据网格 (0=未知, 1=占用, 0.5=可能占用)
        occupancy_grid = np.full((height, width), 0.5, dtype=np.float32)
        
        # 将点云投影到网格
        grid_x = ((x_rel) / resolution).astype(int)
        grid_y = ((y_rel) / resolution).astype(int)
        
        # 过滤有效的网格索引
        valid_mask = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height)
        grid_x = grid_x[valid_mask]
        grid_y = grid_y[valid_mask]
        
        # 标记有点的网格为占用
        # 使用贝叶斯更新：有点增加置信度
        occupancy_grid[grid_y, grid_x] = np.minimum(1.0, occupancy_grid[grid_y, grid_x] + 0.1)
        
        # 平滑处理：如果有网格没有点，从相邻网格推断状态
        # 这里使用简单的空洞填充
        
        logger.info(f"点云 -> 占据网格: {width}x{height}, 分辨率={resolution}m")
        
        return occupancy_grid, {
            "resolution": resolution,
            "width": width,
            "height": height,
            "origin": origin,
            "num_points": len(points)
        }
    
    def compute_color_encoding(
        self, 
        points: np.ndarray,
        mode: str = "height"
    ) -> np.ndarray:
        """
        计算点的颜色编码（用于可视化）
        
        Args:
            points: 点云数组 (N, 3)
            mode: 编码模式 ("height", "distance", "intensity")
            
        Returns:
            colors: 颜色数组 (N, 3) 或 (N, 4)
        """
        if mode == "height":
            # 基于 z 高度编码（蓝色到红色）
            z_values = points[:, 2]
            z_min, z_max = np.min(z_values), np.max(z_values)
            z_range = z_max - z_min
            
            # 归一化到 0-1
            normalized_z = (z_values - z_min) / (z_range + 0.001)
            
            # 使用 HSV 颜色：蓝色(低) -> 绿色(中) -> 红色(高)
            colors = np.zeros((len(points), 3), dtype=np.float32)
            colors[:, 0] = normalized_z  # 红色通道
            colors[:, 1] = 1.0 - normalized_z * 0.5  # 绿色通道
            colors[:, 2] = normalized_z  # 蓝色通道
            
        elif mode == "distance":
            # 基于到原点距离编码
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
            d_min, d_max = np.min(distances), np.max(distances)
            normalized_d = (distances - d_min) / (d_max - d_min + 0.001)
            
            colors = np.zeros((len(points), 3), dtype=np.float32)
            colors[:, 0] = normalized_d
            colors[:, 1] = 1.0 - normalized_d
            colors[:, 2] = normalized_d
            
        else:
            # 默认：统一颜色
            colors = np.ones((len(points), 3), dtype=np.float32) * 0.5
        
        return colors
    
    def save_pointcloud(self, points: np.ndarray, filepath: str, colors: Optional[np.ndarray] = None):
        """
        保存点云到文件
        
        Args:
            points: 点云数组
            filepath: 保存路径
            colors: 可选的颜色数组
        """
        try:
            import open3d as o3d
            import struct
            
            # 创建点云对象
            pcd = o3d.geometry.PointCloud()
            
            if len(points) > 0:
                pcd.points = o3d.utility.Vector3dVector()
                if colors is not None:
                    # 有颜色信息
                    for point, color in zip(points, colors):
                        pcd_point = o3d.geometry.Point3d(
                            x=point[0], y=point[1], z=point[2]
                        )
                        # 转换为 RGB
                        r = int(color[0] * 255)
                        g = int(color[1] * 255)
                        b = int(color[2] * 255)
                        pcd_point.color = o3d.utility.Vector3dVector([r, g, b])
                        pcd.points.append(pcd_point)
                else:
                    # 只有坐标
                    for point in points:
                        pcd.points.append(
                            o3d.geometry.Point3d(x=point[0], y=point[1], z=point[2])
                        )
            
            # 保存
            o3d.io.write_point_cloud(filepath, pcd)
            logger.info(f"点云已保存: {filepath} ({len(points)} 个点)")
            
        except ImportError:
            logger.warning("open3d 不可用，使用 PLY 格式")
            
            # 使用 PLY 格式保存
            header = """ply
format ascii 1.0
element vertex {}
property list uchar float red
property list uchar float green
property list uchar float blue
end_header
"""
            
            vertex_count = len(points)
            with open(filepath, 'w') as f:
                f.write(header)
                
                # 写入顶点数据
                for i, point in enumerate(points):
                    if i < vertex_count:
                        f.write(f"{point[0]} {point[1]} {point[2]}")
                        if colors is not None and i < len(colors):
                            # 写入颜色
                            r = int(colors[i][0] * 255)
                            g = int(colors[i][1] * 255)
                            b = int(colors[i][2] * 255)
                            f.write(f"{r} {g} {b}")
                        else:
                            f.write("255 255 255")  # 白色
                
                f.write("end_header\n")
                f.write(f"end_vertex {vertex_count}\n")
            
            logger.info(f"点云已保存（PLY 格式）: {filepath}")
    
    def process(
        self, 
        pointcloud_msg: Any,
        return_colors: bool = True
    ) -> PointCloudData:
        """
        处理点云消息
        
        Args:
            pointcloud_msg: PointCloud2 消息
            return_colors: 是否返回颜色编码
            
        Returns:
            PointCloudData: 处理后的点云数据
        """
        # 1. 解析点云
        points = self.parse_pointcloud2(pointcloud_msg)
        
        if len(points) == 0:
            logger.warning("点云消息为空")
            return PointCloudData(points=points)
        
        # 2. 应用处理流水线
        self.stats["processed_count"] += 1
        self.stats["total_points"] += len(points)
        
        # 步骤 1: 地面分离
        ground_points, nonground_points = self.remove_ground(points)
        
        # 步骤 2: 噪声过滤
        if self.config.get("noise_filter", True):
            nonground_points = self.remove_noise(nonground_points)
        
        # 步骤 3: 降采样
        downsampled_points = self.downsample(
            nonground_points,
            self.downsample_factor
        )
        
        self.stats["filtered_points"] = len(downsampled_points)
        
        # 4. 计算颜色编码
        colors = None
        if return_colors:
            colors = self.compute_color_encoding(downsampled_points)
        
        # 创建点云数据对象
        pointcloud_data = PointCloudData(
            points=downsampled_points,
            colors=colors
        )
        
        logger.info(
            f"点云处理完成: "
            f"{len(points)} -> {len(downsampled_points)} 点, "
            f"过滤率: {100 * len(downsampled_points) / len(points):.1f}%"
        )
        
        return pointcloud_data
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计"""
        stats = self.stats.copy()
        if stats["processed_count"] > 0:
            stats["avg_points_per_msg"] = stats["total_points"] / stats["processed_count"]
            stats["avg_filtered"] = stats["filtered_points"] / stats["processed_count"]
        
        return stats


def create_processor(config: Optional[Dict[str, Any]] = None) -> LidarProcessor:
    """
    创建点云处理器
    
    Args:
        config: 配置字典
        
    Returns:
        LidarProcessor: 处理器实例
    """
    return LidarProcessor(config)

