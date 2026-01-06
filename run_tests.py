#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接运行认知层测试的脚本 - 绕过ROS2插件问题
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 禁用ROS2的launch_testing插件
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = 'launch-testing'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] += ',launch-testing-ros'

import pytest

def main():
    """运行认知层测试"""
    # 运行所有认知层单元测试
    exit_code = pytest.main([
        'tests/unit/cognitive/',
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
