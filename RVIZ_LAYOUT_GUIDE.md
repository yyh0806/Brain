# RViz2 布局调整指南

## 🚀 启动 RViz2

```bash
cd /media/yangyuhui/CODES1/Brain
python3 launch_rviz2_final.py
```

---

## 📐 手动调整布局（推荐）

### 1. 缩小左侧面板

RViz2 启动后，左侧面板（Displays）可能太大：

1. **找到分割线**
   - 左侧面板（Displays）和中间3D视图之间有一条垂直分割线
   - 鼠标移动到这条线上，光标会变成双向箭头

2. **拖动调整**
   - 左键按住分割线
   - 向左拖动，缩小左侧面板
   - 建议宽度：**200-300 像素**

3. **保存布局**
   - 调整好后，关闭 RViz2
   - 选择保存配置

---

### 2. 增大图像显示

图像默认显示在右下角，可能太小：

1. **打开图像面板**
   - 如果看不到图像，点击菜单栏的 **Panels**
   - 勾选 **RGB Camera**

2. **调整图像面板大小**
   - 找到图像面板周围的分割线
   - 拖动调整图像区域
   - 建议尺寸：**640x400 或更大**

3. **移动图像面板**
   - 右键点击图像面板标题栏
   - 选择 **Detach**（分离）
   - 拖动到任意位置

---

## 🔧 图像不显示的解决方案

### 方法 1：手动添加图像显示

1. 点击左下角的 **Add** 按钮
2. 选择 **Image**
3. 设置参数：
   - **Name**: `RGB Camera`
   - **Image Topic**: `/front_stereo_camera/left/image_raw`
   - **Transport Hint**: `raw`
   - **Normalize Range**: `false`
   - **Min Value**: `0`
   - **Max Value**: `255`
4. 勾选 **Enabled**
5. 点击左下角的 **Image** 按钮打开图像面板

### 方法 2：检查现有图像显示

1. 在左侧 Displays 面板找到 **RGB Camera**
2. 确保已勾选 **Enabled**
3. 展开 **RGB Camera** 查看参数：
   - **Image Topic** 应该是 `/front_stereo_camera/left/image_raw`
   - 如果不对，点击话题选择器重新选择
4. 点击左下角的 **Image** 按钮打开面板

### 方法 3：禁用冲突的显示

如果使用了 `Camera` 显示而不是 `Image`：

1. 在左侧 Displays 面板找到 **Stereo Camera** 或 **Camera**
2. 取消勾选 **Enabled**（禁用）
3. 确保只使用 **Image** 显示

---

## 🎨 最佳布局建议

### 推荐布局

```
┌─────────────────────────────────────────────────────────────┐
│  菜单栏                                                  │
├──────────┬────────────────────────────────────────────────────┤
│          │                                             │
│ Displays │          3D 视图（点云 + 轨迹）              │
│          │                                             │
│          │                                             │
│          │                                             │
│          │                                             │
│          ├───────────────────────────────────────────────────┤
│          │                                             │
│          │          图像面板（RGB Camera）                │
│          │                                             │
│          │                                             │
└──────────┴──────────────────────────────────────────────────┘
```

### 面板大小建议

- **左侧 Displays**: 200-250 像素宽
- **3D 视图**: 占据剩余空间的大部分
- **图像面板**: 右下角，至少 640x400

---

## 📝 显示项说明

### 3D 视图中的显示

1. **PointCloud2** - 3D 点云
   - 彩色点云，按 Z 轴高度编码
   - 点大小：3 像素
   - 颜色：蓝（低）→ 红（高）

2. **Odometry** - 机器人轨迹
   - 绿色箭头
   - 显示机器人位置和方向
   - 保留最近 100 个点

3. **TF** - 坐标系
   - 显示机器人坐标系
   - 只显示 odom 和 base_link

4. **Grid** - 参考网格
   - 灰色网格
   - 10x10m 范围
   - 透明度：0.3

### 图像面板

- **RGB Camera**
  - 实时显示相机图像
  - 分辨率：1920x1200
  - 编码：RGB8

---

## 🖱️ 常用操作

### 3D 视图操作

- **左键拖动**：旋转视图
- **中键拖动**：平移视图
- **滚轮**：缩放（前=放大，后=缩小）
- **双击**：聚焦到物体

### 面板操作

- **拖动分割线**：调整面板大小
- **右键点击分割线**：
  - Hide Left Dock：隐藏左侧面板
  - Hide Right Dock：隐藏右侧面板
  - Split Dock: Left/Right：分割面板
- **拖动面板标题栏**：移动面板位置

---

## 🔍 故障排查

### 问题 1：图像面板是空白的

**检查清单：**
1. 话题是否正确：`/front_stereo_camera/left/image_raw`
2. 图像面板是否打开（点击左下角的 Image 按钮）
3. 启用了正确的显示类型（Image，不是 Camera）
4. Transport Hint 是否为 `raw`

**解决方法：**
- 点击左下角的 **Image** 按钮
- 或在 Panels 菜单勾选 **RGB Camera**

### 问题 2：左侧面板太大

**解决方法：**
- 拖动左侧面板和3D视图之间的分割线
- 向左拖动，缩小面板宽度
- 建议宽度：200-250 像素

### 问题 3：3D 视图太小

**解决方法：**
- 隐藏右侧面板（右键点击分割线 → Hide Right Dock）
- 拖动图像面板和3D视图之间的分割线
- 向下拖动，增大3D视图

### 问题 4：看不到点云

**检查清单：**
1. Fixed Frame 是否为 `odom`（在 Global Options 中）
2. PointCloud2 是否启用（勾选 Enabled）
3. 话题是否正确：`/front_3d_lidar/lidar_points`
4. Isaac Sim 是否在运行且已按 Play

**解决方法：**
- 在 Global Options 中设置 Fixed Frame 为 `odom`
- 等待几秒钟让数据加载

---

## 💾 保存自定义布局

1. 调整好布局后
2. 菜单栏 → **File** → **Save Config As...**
3. 保存为：`config/rviz2/my_custom_layout.rviz`

下次启动时使用：
```bash
rviz2 -d config/rviz2/my_custom_layout.rviz
```

---

## 📚 更多资源

- RViz2 官方文档：https://docs.ros.org/en/galactic/Tutorials/Rviz2-User-Guide.html
- RViz2 显示类型：https://docs.ros.org/en/galactic/Tutorials/Rviz2-Display-Types.html

---

## ⚡ 快捷键

- **Ctrl+Q**：退出 RViz2
- **Ctrl+S**：保存配置
- **Ctrl+L**：加载配置
- **F1**：帮助
- **Ctrl+Space**：切换全屏

---

## 🎯 总结

**最佳实践：**
1. 先调整左侧面板大小（200-250px）
2. 打开图像面板（点击 Image 按钮）
3. 调整图像面板大小（640x400+）
4. 隐藏不必要的面板（右键分割线）
5. 保存自定义布局以备后用

**关键设置：**
- Fixed Frame: `odom`
- Image Topic: `/front_stereo_camera/left/image_raw`
- PointCloud Topic: `/front_3d_lidar/lidar_points`




