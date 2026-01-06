#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行功能测试 - 绕过ROS2插件问题
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 禁用所有ROS2插件
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = 'launch-testing,launch-testing-ros,ament-pep257,ament-flake8,ament-lint,ament-xmllint,ament-copyright'

import pytest

def main():
    """运行功能测试"""
    print("\n" + "="*70)
    print("运行认知层功能测试")
    print("="*70 + "\n")

    # 运行测试
    exit_code = pytest.main([
        'tests/functional/cognitive/',
        '-v',
        '--tb=short',
        '-p', 'asyncio',  # Explicitly load pytest-asyncio
        '-p', 'no:launch_testing',
        '-p', 'no:launch-testing-ros',
        '-p', 'no:ament-pep257',
        '-p', 'no:ament-flake8',
        '-p', 'no:ament-lint',
        '-p', 'no:ament-xmllint',
        '-p', 'no:ament-copyright',
    ])

    return exit_code

if __name__ == '__main__':
    sys.exit(main())
