#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain配置系统使用示例
展示如何使用统一的配置管理系统
"""

import sys
import os

# 添加config目录到Python路径
config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
sys.path.insert(0, config_dir)

from config_loader import load_config, ConfigLoader, ConfigLoadOptions
from validator import validate_config


def example_basic_usage():
    """基础使用示例"""
    print("=== 基础配置使用 ===")

    # 加载默认配置
    config = load_config()

    # 访问系统配置
    system_name = config.get('system', {}).get('name')
    print(f"系统名称: {system_name}")

    # 访问LLM配置
    llm_config = config.get('llm', {})
    print(f"LLM提供商: {llm_config.get('provider')}")
    print(f"LLM模型: {llm_config.get('model')}")


def example_platform_specific():
    """平台特定配置示例"""
    print("\n=== 平台特定配置 ===")

    # 加载UGV平台配置
    ugv_config = load_config(platform="ugv")

    # 访问平台特定参数
    if 'platform' in ugv_config:
        platform = ugv_config['platform']
        print(f"平台名称: {platform.get('name')}")
        print(f"平台类型: {platform.get('type')}")

    # 访问运动学配置
    if 'kinematics' in ugv_config:
        kinematics = ugv_config['kinematics']
        print(f"最大线速度: {kinematics.get('max_linear_speed')} m/s")


def example_environment_specific():
    """环境特定配置示例"""
    print("\n=== 环境特定配置 ===")

    # 加载仿真环境配置
    sim_config = load_config(environment="simulation")

    # 访问仿真配置
    if 'simulation' in sim_config:
        simulation = sim_config['simulation']
        print(f"仿真模式: {simulation.get('mode')}")
        print(f"物理引擎: {simulation.get('physics_engine')}")

    # 访问预设场景
    if 'presets' in sim_config:
        presets = sim_config['presets']
        print(f"可用预设: {list(presets.keys())}")


def example_user_preferences():
    """用户偏好配置示例"""
    print("\n=== 用户偏好配置 ===")

    # 加载用户配置
    user_config = load_config(user="yangyuhui")

    # 访问用户偏好
    if 'preferences' in user_config:
        preferences = user_config['preferences']
        print(f"语言设置: {preferences.get('language')}")
        print(f"提示词风格: {preferences.get('prompt_style')}")

    # 访问个性化调试配置
    if 'debug' in user_config:
        debug = user_config['debug']
        print(f"详细日志: {debug.get('verbose_logging')}")


def example_complete_configuration():
    """完整配置示例"""
    print("\n=== 完整配置加载 ===")

    # 加载所有层级配置
    config = load_config(
        platform="ugv",
        environment="simulation",
        user="yangyuhui"
    )

    # 展示配置层级
    print("配置层级:")
    print(f"  - 系统配置: {'✓' if 'system' in config else '✗'}")
    print(f"  - LLM配置: {'✓' if 'llm' in config else '✗'}")
    print(f"  - 感知配置: {'✓' if 'modules' in config else '✗'}")
    print(f"  - 平台配置: {'✓' if 'platform' in config else '✗'}")
    print(f"  - 仿真配置: {'✓' if 'simulation' in config else '✗'}")
    print(f"  - 用户配置: {'✓' if 'preferences' in config else '✗'}")


def example_config_validation():
    """配置验证示例"""
    print("\n=== 配置验证 ===")

    # 加载配置
    config = load_config()

    # 验证配置
    is_valid, result = validate_config(config)

    if is_valid:
        print("配置验证通过 ✓")
    else:
        print("配置验证失败:")
        print(result)


def example_advanced_loader():
    """高级加载器使用示例"""
    print("\n=== 高级加载器使用 ===")

    # 创建自定义加载选项
    options = ConfigLoadOptions(
        platform="ugv",
        environment="simulation",
        user="yangyuhui",
        merge_strategy="merge",  # 合并策略
        validate_schema=True,    # 启用模式验证
        env_prefix="BRAIN_"      # 环境变量前缀
    )

    # 创建加载器
    loader = ConfigLoader(options)

    # 加载配置
    config = loader.load_config()

    # 获取特定模式
    llm_schema = loader.get_schema("llm")
    if llm_schema:
        print(f"LLM模式定义可用: {list(llm_schema.keys())}")


def example_environment_override():
    """环境变量覆盖示例"""
    print("\n=== 环境变量覆盖 ===")

    # 设置环境变量
    os.environ['BRAIN_LLM__TEMPERATURE'] = '0.9'
    os.environ['BRAIN_LOGGING__LEVEL'] = 'DEBUG'

    # 加载配置
    config = load_config()

    # 检查覆盖效果
    temp = config.get('llm', {}).get('temperature')
    log_level = config.get('logging', {}).get('level')

    print(f"LLM温度 (环境变量覆盖): {temp}")
    print(f"日志级别 (环境变量覆盖): {log_level}")

    # 清理环境变量
    del os.environ['BRAIN_LLM__TEMPERATURE']
    del os.environ['BRAIN_LOGGING__LEVEL']


def example_nested_access():
    """嵌套配置访问示例"""
    print("\n=== 嵌套配置访问 ===")

    config = load_config(platform="ugv", user="yangyuhui")

    # 使用dict访问
    llm_temp = config.get('llm', {}).get('temperature', 0.1)
    print(f"LLM温度: {llm_temp}")

    # 访问模块配置
    perception_config = config.get('modules', {}).get('perception', {})
    sensors = perception_config.get('sensors', {})
    camera_config = sensors.get('camera', {})
    print(f"相机分辨率: {camera_config.get('resolution')}")

    # 访问用户快捷配置
    shortcuts = config.get('shortcuts', {})
    if shortcuts and 'tasks' in shortcuts:
        print(f"可用快捷任务: {list(shortcuts['tasks'].keys())}")


def main():
    """主函数"""
    print("Brain配置系统使用示例")
    print("=" * 50)

    # 运行所有示例
    example_basic_usage()
    example_platform_specific()
    example_environment_specific()
    example_user_preferences()
    example_complete_configuration()
    example_config_validation()
    example_advanced_loader()
    example_environment_override()
    example_nested_access()

    print("\n" + "=" * 50)
    print("示例演示完成！")


if __name__ == "__main__":
    main()