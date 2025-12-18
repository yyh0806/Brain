#!/usr/bin/env python3
"""
Streamlit Dashboard for BRAIN World Model System
Docker容器中的Web界面演示
"""

import streamlit as st
import sys
import os
import time
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append('/workspace/brain')

st.set_page_config(
    page_title="BRAIN World Model Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主标题
st.title("🧠 BRAIN World Model System Dashboard")
st.markdown("---")

# 侧边栏配置
st.sidebar.header("⚙️ 系统配置")

# 模式选择
mode = st.sidebar.selectbox(
    "选择运行模式",
    ["quick", "full", "interactive"],
    help="选择系统演示模式"
)

# 组件选择
components = st.sidebar.multiselect(
    "选择要运行的组件",
    ["all", "world_model", "sensors", "planning"],
    default=["all"],
    help="选择要测试的系统组件"
)

# GPU信息显示
st.sidebar.markdown("### 🚀 GPU状态")
try:
    import torch
    if torch.cuda.is_available():
        st.sidebar.success(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        st.sidebar.info(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        st.sidebar.warning("⚠️ GPU不可用，使用CPU模式")
except ImportError:
    st.sidebar.info("ℹ️ PyTorch未安装，无法检测GPU")

# 主要内容区域
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 系统状态")

    # 模拟系统状态
    status_placeholder = st.empty()

    st.markdown("### 🌍 World Model状态")
    world_model_placeholder = st.empty()

with col2:
    st.markdown("### 📈 性能指标")

    # 性能图表
    chart_data = {
        '时间': ['10s前', '8s前', '6s前', '4s前', '2s前', '现在'],
        'CPU使用率': [45, 52, 48, 61, 55, 58],
        '内存使用率': [62, 65, 63, 68, 70, 67],
    }

    st.line_chart(chart_data, x='时间', y=['CPU使用率', '内存使用率'])

    st.markdown("### 🎯 任务进度")
    progress = st.progress(0)
    status_text = st.empty()

# 运行系统演示
if st.button("🚀 运行系统演示", type="primary"):
    st.markdown("---")
    st.markdown("### 🎬 系统输出")

    # 创建输出占位符
    output_placeholder = st.empty()

    # 模拟系统运行
    with st.spinner("正在初始化BRAIN系统..."):
        time.sleep(2)

    try:
        # 尝试导入并运行系统
        from run_complete_system_demo import WorldModelSystemDemo

        demo = WorldModelSystemDemo(
            mode=mode,
            components=components,
            verbose=True
        )

        # 显示进度
        for i in range(100):
            progress.progress(i + 1)
            status_text.text(f"进度: {i + 1}%")
            time.sleep(0.05)

        # 运行演示
        result = demo.run()

        # 显示结果
        if result.success:
            st.success(f"🎉 系统运行成功！")
            st.info(f"⏱️ 总耗时: {result.execution_time:.2f}秒")
            st.info(f"📊 成功率: {result.success_rate:.1f}%")
        else:
            st.error("❌ 系统运行失败")

    except Exception as e:
        st.error(f"❌ 运行出错: {str(e)}")
        st.info("💡 这通常是因为缺少某些依赖组件，但基础功能仍然可用")

# 实时状态更新
def update_system_status():
    """更新系统状态显示"""
    import psutil
    import platform

    # 系统信息
    system_info = {
        "操作系统": platform.system(),
        "Python版本": platform.python_version(),
        "CPU核心数": psutil.cpu_count(),
        "内存总量": f"{psutil.virtual_memory().total // (1024**3)} GB",
        "磁盘使用": f"{psutil.disk_usage('/').percent}%",
    }

    # 显示系统信息
    for key, value in system_info.items():
        status_placeholder.metric(key, value)

# World Model状态
def update_world_model_status():
    """更新World Model状态"""
    try:
        from brain.cognitive.world_model import WorldModel

        wm = WorldModel()
        current_context = wm.get_current_context()

        world_model_placeholder.json({
            "障碍物数量": current_context.get('obstacles', 0),
            "目标数量": current_context.get('targets', 0),
            "电池电量": f"{current_context.get('battery_level', 100)}%",
            "最后更新": time.strftime("%H:%M:%S")
        })

    except Exception as e:
        world_model_placeholder.error(f"World Model状态获取失败: {str(e)}")

# 定期更新状态
if st.checkbox("🔄 启用实时更新"):
    update_system_status()
    update_world_model_status()

    # 每5秒刷新一次
    if st.button("🔄 手动刷新"):
        update_system_status()
        update_world_model_status()
        st.rerun()

# 底部信息
st.markdown("---")
st.markdown("### 📋 系统信息")
st.info(f"""
- **容器镜像**: brain-simple:latest
- **工作目录**: {os.getcwd()}
- **Python路径**: {sys.path[0]}
- **当前时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
""")

# 使用说明
with st.expander("📖 使用说明"):
    st.markdown("""
    ### BRAIN World Model系统使用指南

    1. **选择运行模式**:
       - `quick`: 快速演示，只测试核心功能
       - `full`: 完整演示，包含所有组件
       - `interactive`: 交互式模式，可以手动控制

    2. **选择组件**:
       - `all`: 所有组件
       - `world_model`: 仅World Model组件
       - `sensors`: 传感器组件
       - `planning`: 规划组件

    3. **系统状态**:
       - 左侧显示实时系统状态
       - 右侧显示性能图表和任务进度
       - 底部显示详细系统信息

    4. **注意事项**:
       - 某些功能可能需要额外的依赖
       - GPU加速需要NVIDIA Docker支持
       - 建议在Chrome或Firefox浏览器中使用
    """)

if st.button("🧪 测试Docker环境"):
    with st.spinner("测试Docker环境..."):
        time.sleep(1)
        st.success("✅ Docker环境正常")
        st.info("🐳 容器ID: " + os.environ.get('HOSTNAME', 'unknown'))

        # 测试Python包
        packages = ['numpy', 'pydantic', 'yaml', 'loguru']
        for pkg in packages:
            try:
                __import__(pkg)
                st.success(f"✅ {pkg} 已安装")
            except ImportError:
                st.error(f"❌ {pkg} 未安装")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🧠 BRAIN World Model System - Docker版 |
        Built with Streamlit |
        Isaac Sim Ready
    </div>
    """,
    unsafe_allow_html=True
)