    def _draw_semantic_info(self, perception_data):
        """绘制语义信息"""
        self.ax_semantic.clear()
        
        info_lines = []
        
        # VLM状态
        if self._vlm_enabled:
            info_lines.append("VLM Service:")
            info_lines.append("  Enabled")
        else:
            info_lines.append("VLM Service:")
            info_lines.append("  Not Enabled")
        
        # 数据接收状态
        info_lines.append("Data Status:")
        info_lines.append(f"  RGB Left: {'✓' if perception_data.rgb_image is not None else '✗'}")
        
        # 右眼RGB状态
        rgb_right = getattr(perception_data, 'rgb_image_right', None)
        if rgb_right is not None:
            info_lines.append(f"  RGB Right: ✓ (Range: [{rgb_right.min()}, {rgb_right.max()}])")
        else:
            info_lines.append("  RGB Right: ✗ (No data)")
        
        # 其他传感器状态
        info_lines.append(f"  Lidar: {'✓' if perception_data.laser_ranges else '✗'}")
        info_lines.append(f"  PointCloud: {'✓' if perception_data.pointcloud is not None else '✗'}")
        info_lines.append(f"  Pose: {'✓' if perception_data.pose else '✗'}")
        
        # 语义物体
        if perception_data.semantic_objects:
            info_lines.append(f"\nSemantic Objects: {len(perception_data.semantic_objects)}")
            # 显示前5个物体信息
            for obj in sorted(perception_data.semantic_objects.values(),
                                  key=lambda x: x.confidence if hasattr(x, 'confidence') else 0,
                                  reverse=True)[:5]:
                label = getattr(obj, 'label', 'unknown')
                conf = getattr(obj, 'confidence', 0.0)
                info_lines.append(f"  • {label} (conf: {conf:.2f})")
        else:
            info_lines.append(f"\nSemantic Objects: None (VLM may not be running yet)")
        
        # 场景描述
        if perception_data.scene_description:
            summary = perception_data.scene_description.summary
            if len(summary) > 50:
                summary = summary[:50] + "..."
            info_lines.append(f"\nScene: {summary}")
        
        # 导航提示
        if perception_data.navigation_hints and len(perception_data.navigation_hints) > 0:
            info_lines.append(f"\nNavigation Hints ({len(perception_data.navigation_hints)}):")
            for hint in perception_data.navigation_hints[:5]:
                info_lines.append(f"  • {hint}")
            if len(perception_data.navigation_hints) > 5:
                info_lines.append(f"  ... and {len(perception_data.navigation_hints)-5} more")
        
        # 世界模型数据
        if hasattr(perception_data, 'world_metadata') and perception_data.world_metadata:
            info_lines.append(f"\nWorld Model:")
            meta = perception_data.world_metadata
            info_lines.append(f"  Updates: {meta.get('update_count', 0)}")
            info_lines.append(f"  Confidence: {meta.get('confidence', 0):.2f}")
        else:
            info_lines.append("")
        
        self.ax_semantic.text(0.02, 0.98, '\n'.join(info_lines),
                            transform=self.ax_semantic.transAxes,
                            verticalalignment='top',
                            fontsize=8,
                            family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        self.ax_semantic.set_title('VLM & World Model Status', fontsize=9)

