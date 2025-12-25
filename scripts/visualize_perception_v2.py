        # 语义物体
        if perception_data.semantic_objects:
            info_lines.append(f"  Semantic Objects: {len(perception_data.semantic_objects)}")
            # 显示最详细的5个物体信息
            for obj in sorted(perception_data.semantic_objects.values(),
                                  key=lambda x: x.confidence if hasattr(x, 'confidence') else 0,
                                  reverse=True)[:5]:
                label = getattr(obj, 'label', 'unknown')
                conf = getattr(obj, 'confidence', 0.0)
                info_lines.append(f"  • {label} (conf: {conf:.2f})")
        else:
            info_lines.append("  Semantic Objects: None")
        info_lines.append("  (VLM may not be running yet)")
        info_lines.append("")
        info_lines.append("")
        info_lines.append("")
        # VLM状态和世界模型统计
        info_lines.append("VLM Service:")
        info_lines.append("  Running" if self._vlm_enabled else "  Not Enabled")

        # 世界模型数据
        info_lines.append("World Model:")
        if hasattr(perception_data, 'world_metadata') and perception_data.world_metadata:
            meta = perception_data.world_metadata
            info_lines.append(f"  Map Updates: {meta.get('update_count', 0)}")
            info_lines.append(f"  Confidence: {meta.get('confidence', 0):.2f}")
            info_lines.append(f"  Created: {meta.get('created_at', 'N/A')}")
            info_lines.append(f"  Last Update: {meta.get('last_updated', 'N/A')}")
        else:
            info_lines.append("  World Model: Not Available")

        self.ax_semantic.text(0.02, 0.98, '\n'.join(info_lines),
                            transform=self.ax_semantic.transAxes,
                            verticalalignment='top',
                            fontsize=8,
                            family='monospace',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        self.ax_semantic.set_title('Semantic Info & World Model', fontsize=9)
