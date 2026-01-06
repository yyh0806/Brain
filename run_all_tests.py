#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有认知层测试的脚本 - 支持异步测试
"""

import sys
import os
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 禁用ROS2插件
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = 'launch-testing,launch-testing-ros,ament-pep257,ament-flake8,ament-lint,ament-xmllint,ament-copyright'

def run_unit_tests():
    """运行单元测试"""
    print("\n" + "="*70)
    print("运行认知层单元测试")
    print("="*70 + "\n")

    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/unit/cognitive/',
        '-v',
        '--tb=short',
        '-p', 'asyncio',
        '-p', 'no:launch_testing',
        '-p', 'no:launch-testing-ros',
        '-p', 'no:ament-pep257',
        '-p', 'no:ament-flake8',
        '-p', 'no:ament-lint',
        '-p', 'no:ament-xmllint',
        '-p', 'no:ament-copyright',
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    return result.returncode

def run_functional_tests():
    """运行功能测试"""
    print("\n" + "="*70)
    print("运行认知层功能测试")
    print("="*70 + "\n")

    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/functional/cognitive/',
        '-v',
        '--tb=short',
        '-p', 'asyncio',
        '-p', 'no:launch_testing',
        '-p', 'no:launch-testing-ros',
        '-p', 'no:ament-pep257',
        '-p', 'no:ament-flake8',
        '-p', 'no:ament-lint',
        '-p', 'no:ament-xmllint',
        '-p', 'no:ament-copyright',
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    return result.returncode

def run_integration_tests():
    """运行集成测试"""
    print("\n" + "="*70)
    print("运行认知层集成测试")
    print("="*70 + "\n")

    result = subprocess.run([
        sys.executable, '-m', 'pytest',
        'tests/integration/test_l3_cognitive.py',
        '-v',
        '--tb=short',
        '-p', 'asyncio',
        '-p', 'no:launch_testing',
        '-p', 'no:launch-testing-ros',
        '-p', 'no:ament-pep257',
        '-p', 'no:ament-flake8',
        '-p', 'no:ament-lint',
        '-p', 'no:ament-xmllint',
        '-p', 'no:ament-copyright',
    ], cwd=os.path.dirname(os.path.abspath(__file__)))

    return result.returncode

def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("认知层完整测试套件")
    print("="*70)

    results = {
        'unit': run_unit_tests(),
        'functional': run_functional_tests(),
        'integration': run_integration_tests(),
    }

    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"单元测试: {'通过 ✓' if results['unit'] == 0 else '失败 ✗'}")
    print(f"功能测试: {'通过 ✓' if results['functional'] == 0 else '失败 ✗'}")
    print(f"集成测试: {'通过 ✓' if results['integration'] == 0 else '失败 ✗'}")
    print("="*70 + "\n")

    return max(results.values())

if __name__ == '__main__':
    sys.exit(main())
