#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置系统测试脚本
验证配置加载、合并和验证功能
"""

import sys
import os
import json
import logging
from pathlib import Path

# 添加config目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from config_loader import ConfigLoader, load_config, ConfigLoadOptions
from validator import ConfigValidator, validate_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_basic_loading():
    """测试基础配置加载"""
    print("\n=== 测试基础配置加载 ===")

    try:
        config = load_config()
        print("✓ 基础配置加载成功")
        print("  系统名称: {}".format(config.get('system', {}).get('name')))
        print("  LLM提供商: {}".format(config.get('llm', {}).get('provider')))
        return True
    except Exception as e:
        print(f"✗ 基础配置加载失败: {e}")
        return False


def test_platform_loading():
    """测试平台特定配置加载"""
    print("\n=== 测试平台特定配置加载 ===")

    try:
        config = load_config(platform="ugv")
        print("✓ UGV平台配置加载成功")

        # 检查平台配置是否被正确加载
        if 'platform' in config or 'platforms' in config:
            print("  ✓ 平台配置存在")
        else:
            print("  ⚠ 平台配置未找到")

        return True
    except Exception as e:
        print(f"✗ UGV平台配置加载失败: {e}")
        return False


def test_environment_loading():
    """测试环境配置加载"""
    print("\n=== 测试环境配置加载 ===")

    try:
        config = load_config(environment="simulation")
        print("✓ 仿真环境配置加载成功")

        # 检查仿真配置
        if 'simulation' in config:
            print(f"  仿真模式: {config['simulation'].get('mode')}")

        return True
    except Exception as e:
        print(f"✗ 仿真环境配置加载失败: {e}")
        return False


def test_user_loading():
    """测试用户配置加载"""
    print("\n=== 测试用户配置加载 ===")

    try:
        config = load_config(user="yangyuhui")
        print("✓ 用户配置加载成功")

        # 检查用户个性化配置
        if 'preferences' in config:
            print(f"  用户语言: {config['preferences'].get('language')}")

        return True
    except Exception as e:
        print(f"✗ 用户配置加载失败: {e}")
        return False


def test_full_loading():
    """测试完整配置加载（所有层级）"""
    print("\n=== 测试完整配置加载 ===")

    try:
        config = load_config(
            platform="ugv",
            environment="simulation",
            user="yangyuhui"
        )
        print("✓ 完整配置加载成功")

        # 验证各层级配置
        print("  配置层级检查:")
        print(f"    - 系统配置: {'✓' if 'system' in config else '✗'}")
        print(f"    - LLM配置: {'✓' if 'llm' in config else '✗'}")
        print(f"    - 感知配置: {'✓' if 'perception' in config or 'modules' in config else '✗'}")
        print(f"    - 平台配置: {'✓' if 'platform' in config else '✗'}")
        print(f"    - 环境配置: {'✓' if 'simulation' in config else '✗'}")
        print(f"    - 用户配置: {'✓' if 'preferences' in config else '✗'}")

        return True
    except Exception as e:
        print(f"✗ 完整配置加载失败: {e}")
        return False


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")

    try:
        # 加载一个配置
        config = load_config(platform="ugv")

        # 验证配置
        is_valid, result = validate_config(config)

        if is_valid:
            print("✓ 配置验证通过")
        else:
            print("⚠ 配置验证发现问题:")
            print(result)

        return True
    except Exception as e:
        print(f"✗ 配置验证失败: {e}")
        return False


def test_invalid_config():
    """测试无效配置的验证"""
    print("\n=== 测试无效配置验证 ===")

    try:
        # 创建一个无效配置
        invalid_config = {
            "llm": {
                "provider": "invalid_provider",
                "temperature": 5.0,  # 超出范围
                "model": ""
            },
            "perception": {
                "update_rate": -1  # 无效值
            }
        }

        # 验证配置
        is_valid, errors = validate_config(invalid_config)

        if not is_valid:
            print("✓ 成功检测到配置错误")
            print(f"  错误数量: {len(errors.split(chr(10))) - 1}")
            return True
        else:
            print("✗ 未能检测到配置错误")
            return False

    except Exception as e:
        print(f"✗ 无效配置测试失败: {e}")
        return False


def test_config_priority():
    """测试配置优先级"""
    print("\n=== 测试配置优先级 ===")

    try:
        # 加载不同层级的配置
        base_config = load_config()
        user_config = load_config(user="yangyuhui")

        # 检查用户配置是否覆盖了基础配置
        base_temp = base_config.get('llm', {}).get('temperature', 0.1)
        user_temp = user_config.get('llm', {}).get('temperature', 0.1)

        # 检查日志级别覆盖
        base_log_level = base_config.get('logging', {}).get('level', 'INFO')
        user_log_level = user_config.get('logging', {}).get('level', 'INFO')

        print(f"  基础配置温度: {base_temp}")
        print(f"  用户配置温度: {user_temp}")
        print(f"  基础日志级别: {base_log_level}")
        print(f"  用户日志级别: {user_log_level}")

        if user_log_level != base_log_level:
            print("✓ 用户配置成功覆盖基础配置")
            return True
        else:
            print("⚠ 配置覆盖可能未正常工作")
            return False

    except Exception as e:
        print(f"✗ 配置优先级测试失败: {e}")
        return False


def test_environment_variable_override():
    """测试环境变量覆盖"""
    print("\n=== 测试环境变量覆盖 ===")

    try:
        # 设置环境变量
        os.environ['BRAIN_LLM__TEMPERATURE'] = '0.8'
        os.environ['BRAIN_LOGGING__LEVEL'] = 'DEBUG'

        # 加载配置
        config = load_config()

        # 检查是否被环境变量覆盖
        temp = config.get('llm', {}).get('temperature', None)
        log_level = config.get('logging', {}).get('level', None)

        print(f"  温度: {temp}")
        print(f"  日志级别: {log_level}")

        # 清理环境变量
        del os.environ['BRAIN_LLM__TEMPERATURE']
        del os.environ['BRAIN_LOGGING__LEVEL']

        if temp == 0.8 and log_level == 'DEBUG':
            print("✓ 环境变量覆盖成功")
            return True
        else:
            print("⚠ 环境变量覆盖可能未正常工作")
            return False

    except Exception as e:
        print(f"✗ 环境变量覆盖测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("Brain配置系统测试")
    print("=" * 50)

    tests = [
        test_basic_loading,
        test_platform_loading,
        test_environment_loading,
        test_user_loading,
        test_full_loading,
        test_config_validation,
        test_invalid_config,
        test_config_priority,
        test_environment_variable_override
    ]

    passed = 0
    total = len(tests)

    for test_func in tests:
        if test_func():
            passed += 1

    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")

    if passed == total:
        print("✓ 所有测试通过！配置系统工作正常。")
        return 0
    else:
        print("⚠ 部分测试失败，请检查配置系统。")
        return 1


if __name__ == "__main__":
    sys.exit(main())