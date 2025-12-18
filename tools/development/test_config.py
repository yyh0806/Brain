#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置系统测试脚本
"""

import sys
import os

# 添加config目录到Python路径
config_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, config_dir)

try:
    from config_loader import load_config
    print "成功导入配置加载器"
except ImportError as e:
    print "导入配置加载器失败: {}".format(e)
    sys.exit(1)

def test_basic():
    """基础测试"""
    print "\n=== 基础配置加载测试 ==="
    try:
        config = load_config()
        print "配置加载成功"

        # 检查基本配置项
        if 'system' in config:
            system_name = config['system'].get('name', 'N/A')
            print "系统名称: {}".format(system_name)

        if 'llm' in config:
            llm_provider = config['llm'].get('provider', 'N/A')
            print "LLM提供商: {}".format(llm_provider)

        return True
    except Exception as e:
        print "配置加载失败: {}".format(e)
        return False

def test_platform():
    """平台配置测试"""
    print "\n=== 平台配置测试 ==="
    try:
        config = load_config(platform="ugv")
        print "UGV平台配置加载成功"
        return True
    except Exception as e:
        print "UGV平台配置加载失败: {}".format(e)
        return False

def test_user():
    """用户配置测试"""
    print "\n=== 用户配置测试 ==="
    try:
        config = load_config(user="yangyuhui")
        print "用户配置加载成功"
        return True
    except Exception as e:
        print "用户配置加载失败: {}".format(e)
        return False

def test_full():
    """完整配置测试"""
    print "\n=== 完整配置测试 ==="
    try:
        config = load_config(
            platform="ugv",
            environment="simulation",
            user="yangyuhui"
        )
        print "完整配置加载成功"

        # 检查各层级配置
        checks = [
            ('系统配置', 'system'),
            ('LLM配置', 'llm'),
            ('仿真配置', 'simulation'),
        ]

        for name, key in checks:
            if key in config:
                print "{}存在".format(name)
            else:
                print "{}未找到".format(name)

        return True
    except Exception as e:
        print "完整配置加载失败: {}".format(e)
        return False

def main():
    """主测试"""
    print "Brain配置系统测试"
    print "=" * 40

    tests = [test_basic, test_platform, test_user, test_full]
    passed = 0

    for test in tests:
        if test():
            passed += 1

    print "\n" + "=" * 40
    print "测试结果: {}/{} 通过".format(passed, len(tests))

    if passed == len(tests):
        print "所有测试通过！"
        return 0
    else:
        print "部分测试失败"
        return 1

if __name__ == "__main__":
    sys.exit(main())