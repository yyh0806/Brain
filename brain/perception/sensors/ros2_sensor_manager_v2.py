    async def get_fused_perception(self) -> PerceptionData:
        """
        获取融合后的感知数据（集成世界模型版本）
        
        Returns:
            PerceptionData: 融合后的感知数据，包含传感器实时数据和全局地图快照
        """
        # 从ROS2接口获取原始数据
        raw_data = self.ros2.get_sensor_data()
        
        # 创建感知数据
        perception = PerceptionData(timestamp=raw_data.timestamp)
        
        # 处理位姿
        if raw_data.odometry:
            perception.pose = self._extract_pose(raw_data.odometry)
            perception.velocity = self._extract_velocity(raw_data.odometry)
            self._update_sensor_status(SensorType.ODOMETRY, True)
        
        # 处理IMU（用于姿态融合）
        if raw_data.imu:
            if perception.pose:
                perception.pose = self._fuse_imu_pose(perception.pose, raw_data.imu)
            self._update_sensor_status(SensorType.IMU, True)
        
        # 处理RGB图像（左眼）
        if raw_data.rgb_image is not None:
            perception.rgb_image = raw_data.rgb_image
            self._update_sensor_status(SensorType.RGB_CAMERA, True)
        
        # 处理RGB图像（右眼）
        if hasattr(raw_data, 'rgb_image_right') and raw_data.rgb_image_right is not None:
            perception.rgb_image_right = raw_data.rgb_image_right
        
        # 处理深度图像
        if raw_data.depth_image is not None:
            perception.depth_image = raw_data.depth_image
            self._update_sensor_status(SensorType.DEPTH_CAMERA, True)
        
        # 处理激光雷达
        if raw_data.laser_scan:
            perception.laser_ranges = raw_data.laser_scan.get("ranges", [])
            perception.laser_angles = self._generate_laser_angles(
                raw_data.laser_scan.get("angle_min", -3.14),
                raw_data.laser_scan.get("angle_max", 3.14),
                len(perception.laser_ranges)
            )
            # 从激光雷达检测障碍物
            perception.obstacles = self._detect_obstacles_from_laser(
                perception.laser_ranges,
                perception.laser_angles,
                perception.pose
            )
            self._update_sensor_status(SensorType.LIDAR, True)
        
        # 处理点云
        if raw_data.pointcloud is not None and raw_data.pointcloud.size > 0:
            perception.pointcloud = raw_data.pointcloud
            self._update_sensor_status(SensorType.LIDAR, True)
            
            # 如果没有激光雷达数据，将点云转换为激光雷达格式
            if perception.laser_ranges is None or len(perception.laser_ranges) == 0:
                laser_data = self._convert_pointcloud_to_laser(
                    raw_data.pointcloud,
                    perception.pose
                )
                if laser_data:
                    perception.laser_ranges = laser_data["ranges"]
                    perception.laser_angles = laser_data["angles"]
                    # 从转换后的激光雷达检测障碍物
                    perception.obstacles = self._detect_obstacles_from_laser(
                        perception.laser_ranges,
                        perception.laser_angles,
                        perception.pose
                    )
        
        # 更新全局世界模型（持久地图）
        if self.world_model:
            self.world_model.update_with_perception(perception)
        
        # 从世界模型获取地图快照（持久、融合的环境表示）
        if self.world_model:
            perception.global_map = self.world_model.get_global_map()
            perception.semantic_objects = self.world_model.semantic_objects.values()
            perception.scene_description = self.world_model.metadata.scene_description if hasattr(self.world_model.metadata, 'scene_description') else None
            perception.spatial_relations = self.world_model.spatial_relations.copy()
            perception.navigation_hints = self.world_model.metadata.navigation_hints if hasattr(self.world_model.metadata, 'navigation_hints') else []
            perception.world_metadata = {
                "created_at": self.world_model.metadata.created_at.isoformat(),
                "last_updated": self.world_model.metadata.last_updated.isoformat(),
                "update_count": self.world_model.metadata.update_count,
                "confidence": self.world_model.metadata.confidence,
                "map_stats": self.world_model.get_map_statistics()
            }
        
        # 触发异步VLM分析（不阻塞主循环）
        if self._vlm_service and raw_data.rgb_image is not None:
            asyncio.create_task(self._trigger_vlm_analysis(raw_data.rgb_image))
        
        # 缓存数据（使用循环缓冲区，自动限制大小）
        self._latest_data = perception
        self._data_history.append(perception)  # deque会自动移除最旧的元素
        
        return perception



