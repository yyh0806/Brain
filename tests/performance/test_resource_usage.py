"""
资源使用测试

测试Brain感知模块的资源使用情况，包括内存使用、CPU使用和资源泄漏检测。
"""

import pytest
import asyncio
import psutil
import gc
import os
import time
import numpy as np
import threading
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ResourceSnapshot:
    """资源快照"""
    timestamp: float
    memory_mb: float
    cpu_percent: float
    open_files: int
    threads_count: int


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_snapshot = self.take_snapshot()
        self.snapshots: List[ResourceSnapshot] = []
    
    def take_snapshot(self) -> ResourceSnapshot:
        """获取资源快照"""
        return ResourceSnapshot(
            timestamp=time.time(),
            memory_mb=self.process.memory_info().rss / 1024 / 1024,
            cpu_percent=self.process.cpu_percent(interval=1),
            open_files=len(self.process.open_files()),
            threads_count=self.process.num_threads()
        )
    
    def record_snapshot(self):
        """记录资源快照"""
        snapshot = self.take_snapshot()
        self.snapshots.append(snapshot)
        
        # 保持最近1000个快照
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]
    
    def get_peak_memory(self) -> float:
        """获取峰值内存使用"""
        if not self.snapshots:
            return 0.0
        return max(snapshot.memory_mb for snapshot in self.snapshots)
    
    def get_peak_cpu(self) -> float:
        """获取峰值CPU使用率"""
        if not self.snapshots:
            return 0.0
        return max(snapshot.cpu_percent for snapshot in self.snapshots)
    
    def get_memory_growth(self) -> float:
        """获取内存增长率（MB/小时）"""
        if len(self.snapshots) < 2:
            return 0.0
        
        first_snapshot = self.snapshots[0]
        last_snapshot = self.snapshots[-1]
        
        time_diff = last_snapshot.timestamp - first_snapshot.timestamp
        if time_diff <= 0:
            return 0.0
        
        memory_diff = last_snapshot.memory_mb - first_snapshot.memory_mb
        return memory_diff / time_diff * 3600  # MB/小时
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        if not self.snapshots:
            return {}
        
        memory_usage = [snapshot.memory_mb for snapshot in self.snapshots]
        cpu_usage = [snapshot.cpu_percent for snapshot in self.snapshots]
        
        return {
            "initial_memory_mb": self.initial_snapshot.memory_mb,
            "current_memory_mb": self.snapshots[-1].memory_mb,
            "memory_growth_mb_per_hour": self.get_memory_growth(),
            "peak_memory_mb": self.get_peak_memory(),
            "avg_memory_mb": sum(memory_usage) / len(memory_usage),
            "min_memory_mb": min(memory_usage),
            "max_memory_mb": max(memory_usage),
            "current_cpu_percent": self.snapshots[-1].cpu_percent,
            "peak_cpu_percent": self.get_peak_cpu(),
            "avg_cpu_percent": sum(cpu_usage) / len(cpu_usage),
            "min_cpu_percent": min(cpu_usage),
            "max_cpu_percent": max(cpu_usage),
            "samples_count": len(self.snapshots),
            "monitoring_duration_minutes": (self.snapshots[-1].timestamp - self.initial_snapshot.timestamp) / 60
        }


class MockROS2Interface:
    """模拟ROS2接口"""
    def __init__(self):
        pass
    
    async def get_sensor_data(self):
        """模拟传感器数据"""
        # 生成较大的数据来测试内存使用
        return Mock(
            timestamp=time.time(),
            rgb_image=np.random.randint(0, 256, (720, 1280, 3)),  # 大图像
            depth_image=np.random.rand(720, 1280) * 10.0,  # 大深度图
            laser_scan={"ranges": [5.0] * 1440, "angles": [i * 0.025 for i in range(1440)]},
            imu={
                "orientation": {"x": 0, "y": 0, "z": 0.1, "w": 0.995},
                "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.1},
                "linear_acceleration": {"x": 0.1, "y": 0.05, "z": 9.81}
            },
            odometry={
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                "linear_velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0, "y": 0, "z": 0.1}
            },
            pointcloud=np.random.rand(10000, 6)  # 大点云
        )
    
    def get_rgb_image(self):
        """获取RGB图像"""
        return np.random.randint(0, 256, (720, 1280, 3))
    
    def get_depth_image(self):
        """获取深度图像"""
        return np.random.rand(720, 1280) * 10.0
    
    def get_laser_scan(self):
        """获取激光雷达数据"""
        return {"ranges": [5.0] * 1440, "angles": [i * 0.025 for i in range(1440)]}
    
    def destroy(self):
        """销毁接口"""
        pass


@pytest.fixture
def resource_monitor():
    """资源监控器"""
    return ResourceMonitor()


@pytest.fixture
async def ros2_interface():
    """模拟ROS2接口"""
    return MockROS2Interface()


@pytest.fixture
async def sensor_manager(ros2_interface, resource_monitor):
    """传感器管理器"""
    # 模拟导入
    with patch('brain.communication.ros2_interface.ROS2Interface', return_value=ros2_interface):
        with patch('brain.perception.sensors.ros2_sensor_manager.OccupancyMapper') as MockOccupancyMapper:
            # 模拟占据栅格映射器
            mock_mapper = Mock()
            mock_mapper.get_grid.return_value = Mock(
                data=np.random.randint(-1, 2, (1000, 1000)),
                resolution=0.1,
                origin_x=-50.0,
                origin_y=-50.0
            )
            MockOccupancyMapper.return_value = mock_mapper
            
            # 导入并创建传感器管理器
            from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
            
            # 创建大缓冲区配置来测试内存使用
            config = {
                "sensors": {
                    "rgb_camera": {"enabled": True},
                    "depth_camera": {"enabled": True},
                    "lidar": {"enabled": True},
                    "imu": {"enabled": True}
                },
                "grid_resolution": 0.05,  # 更高分辨率
                "map_size": 100.0,     # 更大地图
                "max_history": 1000       # 大缓冲区
            }
            
            manager = ROS2SensorManager(ros2_interface, config)
            
            # 开始监控资源
            resource_monitor.record_snapshot()
            
            return manager


class TestMemoryUsage:
    """测试内存使用情况"""
    
    @pytest.mark.asyncio
    async def test_sensor_data_memory_usage(self, sensor_manager, resource_monitor):
        """测试传感器数据获取的内存使用"""
        initial_memory = resource_monitor.snapshots[-1].memory_mb
        peak_memory = initial_memory
        
        # 连续获取传感器数据
        for i in range(50):
            perception_data = await sensor_manager.get_fused_perception()
            
            # 记录内存使用
            resource_monitor.record_snapshot()
            current_memory = resource_monitor.snapshots[-1].memory_mb
            peak_memory = max(peak_memory, current_memory)
        
            # 验证内存使用
            assert current_memory > initial_memory  # 内存应该增长
            assert current_memory < initial_memory + 500  # 但不应增长太多
            
            # 短暂延迟，模拟正常处理间隔
            await asyncio.sleep(0.01)
        
        # 清理内存
        del perception_data
        gc.collect()
        
        # 等待一段时间检查内存是否释放
        await asyncio.sleep(0.1)
        gc.collect()
        
        final_memory = resource_monitor.take_snapshot().memory_mb
        
        # 验证内存清理
        # 由于Python的垃圾回收，内存可能不会立即回到初始值
        # 但应该有所下降
        memory_drop = peak_memory - final_memory
        assert memory_drop > initial_memory * 0.1  # 至少下降10%
        
        stats = resource_monitor.get_resource_stats()
        
        # 验证内存统计
        assert stats["memory_growth_mb_per_hour"] < 1000  # 内存增长率应合理
        assert stats["peak_memory_mb"] < initial_memory + 2000  # 峰值内存应合理
    
    @pytest.mark.asyncio
    async def test_large_data_buffer_memory_usage(self, sensor_manager, resource_monitor):
        """测试大数据缓冲区的内存使用"""
        initial_memory = resource_monitor.snapshots[-1].memory_mb
        
        # 填充数据缓冲区
        for _ in range(100):
            await sensor_manager.get_fused_perception()
            await asyncio.sleep(0.001)  # 短暂延迟
        
        peak_memory = resource_monitor.get_peak_memory()
        
        # 验证内存使用
        assert peak_memory > initial_memory
        
        # 验证内存增长率
        memory_growth = resource_monitor.get_memory_growth()
        assert memory_growth < 5000  # 内存增长率应合理
        
        # 清理缓冲区
        # 由于无法直接访问缓冲区，我们重新创建管理器
        del sensor_manager
        gc.collect()
        
        # 等待垃圾回收
        await asyncio.sleep(0.5)
        
        final_memory = resource_monitor.take_snapshot().memory_mb
        
        # 验证内存释放
        assert final_memory < peak_memory
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, sensor_manager, resource_monitor):
        """测试内存泄漏检测"""
        initial_memory = resource_monitor.snapshots[-1].memory_mb
        
        # 运行长时间测试以检测内存泄漏
        test_duration = 10  # 10秒
        update_rate = 20  # 20Hz
        
        start_time = time.time()
        memory_readings = []
        
        while time.time() - start_time < test_duration:
            # 获取感知数据
            perception_data = await sensor_manager.get_fused_perception()
            
            # 定期记录内存使用
            if len(memory_readings) % 10 == 0:
                resource_monitor.record_snapshot()
                memory_readings.append(resource_monitor.snapshots[-1].memory_mb)
            
            await asyncio.sleep(1.0 / update_rate)
        
        # 分析内存趋势
        if len(memory_readings) > 1:
            first_memory = memory_readings[0]
            last_memory = memory_readings[-1]
            memory_growth = last_memory - first_memory
            
            # 验证内存泄漏检测
            # 内存增长不应过大（考虑缓冲区填充）
            assert memory_growth < first_memory * 0.5  # 增长不应超过初始内存的50%
        
        stats = resource_monitor.get_resource_stats()
        assert stats["memory_growth_mb_per_hour"] < 10000  # 长期内存增长率应合理


class TestCPUUsage:
    """测试CPU使用情况"""
    
    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self, sensor_manager, resource_monitor):
        """测试高负载下的CPU使用"""
        initial_cpu = resource_monitor.snapshots[-1].cpu_percent
        cpu_readings = []
        
        # 运行高强度测试
        test_duration = 5  # 5秒
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # 快速获取感知数据
            perception_data = await sensor_manager.get_fused_perception()
            
            # 模拟一些计算密集型处理
            if perception_data.rgb_image is not None:
                # 对图像进行一些处理
                processed_image = np.mean(perception_data.rgb_image)
                processed_image = np.std(perception_data.rgb_image)
            
            if perception_data.laser_ranges is not None:
                # 对激光数据进行一些处理
                processed_laser = np.mean(perception_data.laser_ranges)
                processed_laser = np.max(perception_data.laser_ranges)
            
            # 定期记录CPU使用
            if len(cpu_readings) % 20 == 0:
                resource_monitor.record_snapshot()
                cpu_readings.append(resource_monitor.snapshots[-1].cpu_percent)
            
            # 短暂延迟
            await asyncio.sleep(0.01)
        
        # 验证CPU使用
        if cpu_readings:
            avg_cpu = sum(cpu_readings) / len(cpu_readings)
            peak_cpu = max(cpu_readings)
            
            # CPU使用应该是合理的
            assert avg_cpu < 90  # 平均CPU使用不应超过90%
            assert peak_cpu < 95  # 峰值CPU使用不应超过95%
            
            # CPU使用应该高于空闲状态
            assert avg_cpu > initial_cpu + 10  # 应该有明显CPU使用
        
        stats = resource_monitor.get_resource_stats()
        assert stats["peak_cpu_percent"] < 100  # 峰值CPU使用不应超过100%
    
    @pytest.mark.asyncio
    async def test_cpu_efficiency(self, sensor_manager, resource_monitor):
        """测试CPU效率"""
        start_time = time.time()
        operations_completed = 0
        
        # 测量操作效率
        test_duration = 3  # 3秒
        
        while time.time() - start_time < test_duration:
            # 获取感知数据
            perception_data = await sensor_manager.get_fused_perception()
            if perception_data is not None:
                operations_completed += 1
            
            await asyncio.sleep(0.001)  # 1ms延迟
        
        elapsed_time = time.time() - start_time
        operations_per_second = operations_completed / elapsed_time
        
        # 验证效率
        assert operations_per_second > 100  # 应该能处理超过100次操作/秒
        assert operations_per_second < 1000  # 但不应该过高（可能表示处理过于简单）


class TestResourceLeaks:
    """测试资源泄漏"""
    
    @pytest.mark.asyncio
    async def test_sensor_manager_resource_leak(self, sensor_manager, resource_monitor):
        """测试传感器管理器资源泄漏"""
        initial_memory = resource_monitor.snapshots[-1].memory_mb
        
        # 多次创建和销毁传感器管理器
        for i in range(5):
            # 创建新的传感器管理器
            with patch('brain.communication.ros2_interface.ROS2Interface', MockROS2Interface):
                with patch('brain.perception.sensors.ros2_sensor_manager.OccupancyMapper'):
                    from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
                    
                    new_manager = ROS2SensorManager(
                        MockROS2Interface(),
                        {"sensors": {"rgb_camera": {"enabled": True}}}
                    )
                    
                    # 使用管理器
                    await new_manager.get_fused_perception()
                    await new_manager.get_fused_perception()
                    
                    # 显式销毁
                    del new_manager
        
            # 垃圾回收
            gc.collect()
            await asyncio.sleep(0.1)
        
        # 验证内存使用
        final_memory = resource_monitor.take_snapshot().memory_mb
        memory_increase = final_memory - initial_memory
        
        # 内存增加应该是合理的
        assert memory_increase < initial_memory * 0.2  # 不应超过初始内存的20%
    
    @pytest.mark.asyncio
    async def test_long_running_resource_usage(self, sensor_manager, resource_monitor):
        """测试长时间运行的资源使用"""
        initial_memory = resource_monitor.snapshots[-1].memory_mb
        initial_snapshot = resource_monitor.take_snapshot()
        
        # 长时间运行测试
        test_duration = 30  # 30秒
        start_time = time.time()
        stable_period_start = 10  # 前10秒不稳定
        memory_samples = []
        
        while time.time() - start_time < test_duration:
            # 获取感知数据
            perception_data = await sensor_manager.get_fused_perception()
            
            # 在稳定期记录内存样本
            if time.time() - start_time > stable_period_start:
                resource_monitor.record_snapshot()
                current_memory = resource_monitor.snapshots[-1].memory_mb
                memory_samples.append(current_memory)
            
            await asyncio.sleep(0.1)
        
        # 分析稳定性期间的内存使用
        if len(memory_samples) > 10:
            avg_memory = sum(memory_samples) / len(memory_samples)
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            memory_variance = sum((m - avg_memory) ** 2 for m in memory_samples) / len(memory_samples)
            memory_std = memory_variance ** 0.5
            
            # 验证内存稳定性
            memory_stability = memory_std / avg_memory
            assert memory_stability < 0.1  # 标准差不应超过平均值的10%
            
            # 内存波动应该合理
            memory_range = max_memory - min_memory
            assert memory_range < avg_memory * 0.2  # 范围不应超过平均值的20%
        
        # 验证总体内存使用
        final_memory = resource_monitor.take_snapshot().memory_mb
        total_increase = final_memory - initial_memory
        
        # 长时间运行的内存增长应该是合理的
        assert total_increase < initial_memory * 0.5  # 不应超过初始内存的50%


def print_resource_results(resource_monitor: ResourceMonitor):
    """打印资源测试结果"""
    stats = resource_monitor.get_resource_stats()
    
    print("\\n资源使用测试结果:")
    print(f"初始内存: {stats.get('initial_memory_mb', 0):.2f} MB")
    print(f"当前内存: {stats.get('current_memory_mb', 0):.2f} MB")
    print(f"内存增长率: {stats.get('memory_growth_mb_per_hour', 0):.2f} MB/小时")
    print(f"峰值内存: {stats.get('peak_memory_mb', 0):.2f} MB")
    print(f"平均内存: {stats.get('avg_memory_mb', 0):.2f} MB")
    
    print(f"当前CPU: {stats.get('current_cpu_percent', 0):.2f}%")
    print(f"峰值CPU: {stats.get('peak_cpu_percent', 0):.2f}%")
    print(f"平均CPU: {stats.get('avg_cpu_percent', 0):.2f}%")
    
    print(f"监控持续时间: {stats.get('monitoring_duration_minutes', 0):.2f} 分钟")
    print(f"样本数量: {stats.get('samples_count', 0)}")


if __name__ == "__main__":
    import sys
    
    # 创建资源监控器
    monitor = ResourceMonitor()
    
    # 运行一个简单的资源监控测试
    async def simple_test():
        mock_interface = MockROS2Interface()
        
        with patch('brain.communication.ros2_interface.ROS2Interface', return_value=mock_interface):
            with patch('brain.perception.sensors.ros2_sensor_manager.OccupancyMapper'):
                from brain.perception.sensors.ros2_sensor_manager import ROS2SensorManager
                
                manager = ROS2SensorManager(mock_interface, {})
                
                # 监控资源
                monitor.record_snapshot()
                
                # 运行一些感知操作
                for _ in range(10):
                    await manager.get_fused_perception()
                    monitor.record_snapshot()
                    await asyncio.sleep(0.01)
                
                print_resource_results(monitor)
                
                return True
    
    try:
        asyncio.run(simple_test())
    except KeyboardInterrupt:
        print("\\n测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\\n测试出错: {e}")
        sys.exit(1)


