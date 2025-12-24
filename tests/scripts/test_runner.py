#!/usr/bin/env python3
"""
测试运行器

这个脚本用于运行Brain感知模块的各种测试，包括单元测试、集成测试和端到端测试。
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, cwd=None, timeout=300):
    """运行命令并返回结果"""
    print(f"运行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False
        )
        
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "命令执行超时"
    except Exception as e:
        return -1, "", f"命令执行失败: {e}"


def run_unit_tests():
    """运行单元测试"""
    print("运行单元测试...")
    
    cmd = ["python", "-m", "pytest", 
           "tests/unit/", 
           "--cov=brain.perception",
           "--cov-report=html",
           "--cov-report=term-missing"]
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode != 0:
        print(f"单元测试失败 (返回码: {returncode})")
        print(f"错误: {stderr}")
        return False
    
    print("单元测试成功完成")
    if stdout:
        print(f"输出: {stdout}")
    
    return True


def run_integration_tests():
    """运行集成测试"""
    print("运行集成测试...")
    
    cmd = ["python", "-m", "pytest", 
           "tests/integration/", 
           "--cov=brain.perception",
           "--cov-report=html",
           "--cov-report=term-missing"]
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode != 0:
        print(f"集成测试失败 (返回码: {returncode})")
        print(f"错误: {stderr}")
        return False
    
    print("集成测试成功完成")
    if stdout:
        print(f"输出: {stdout}")
    
    return True


def run_performance_tests():
    """运行性能测试"""
    print("运行性能测试...")
    
    cmd = ["python", "-m", "pytest", 
           "tests/performance/",
           "--cov=brain.perception",
           "--cov-report=html",
           "--cov-report=term-missing"]
    
    returncode, stdout, stderr = run_command(cmd)
    
    if returncode != 0:
        print(f"性能测试失败 (返回码: {returncode})")
        print(f"错误: {stderr}")
        return False
    
    print("性能测试成功完成")
    if stdout:
        print(f"输出: {stdout}")
    
    return True


def run_isaac_sim_tests():
    """运行Isaac Sim相关测试"""
    print("运行Isaac Sim测试...")
    
    # 检查Isaac Sim是否可用
    check_cmd = ["python", "-c", 
                    "try:\n    import omni.kit.commands\n    print('Isaac Sim Python API可用')\nexcept ImportError:\n    print('Isaac Sim Python API不可用')\n    exit(1)"]
    
    returncode, stdout, stderr = run_command(check_cmd)
    
    if returncode != 0 or "Isaac Sim Python API不可用" in stdout:
        print("警告: Isaac Sim不可用，跳过相关测试")
        return True
    
    print("Isaac Sim可用，开始测试场景创建...")
    
    # 创建测试场景
    scene_cmd = ["python", "scripts/create_isaac_sim_scene.py", "--scene", "simple"]
    returncode, stdout, stderr = run_command(scene_cmd)
    
    if returncode != 0:
        print(f"场景创建失败 (返回码: {returncode})")
        print(f"错误: {stderr}")
        return False
    
    print("场景创建成功")
    print(stdout)
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("运行所有测试...")
    
    results = {
        "unit": run_unit_tests(),
        "integration": run_integration_tests(),
        "performance": run_performance_tests(),
        "isaac_sim": run_isaac_sim_tests()
    }
    
    # 统计结果
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    print(f"\\n测试完成统计: {success_count}/{total_count} 成功")
    
    for test_type, success in results.items():
        status = "成功" if success else "失败"
        print(f"  {test_type}: {status}")
    
    return success_count == total_count


def generate_test_report():
    """生成测试报告"""
    print("生成测试报告...")
    
    # 获取覆盖率报告
    coverage_cmd = ["python", "-c", 
                       "import subprocess\nresult = subprocess.run(['coverage', 'report'], capture_output=True)\nprint(result.stdout)"]
    
    returncode, stdout, stderr = run_command(coverage_cmd)
    
    if returncode == 0:
        print("测试报告生成成功")
        print(stdout)
    else:
        print(f"测试报告生成失败: {stderr}")
    
    return returncode == 0


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Brain感知模块测试运行器")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "isaac", "all"], 
                       default="all", help="测试类型")
    parser.add_argument("--report", action="store_true", 
                       help="生成测试报告")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # 根据参数运行测试
    if args.type == "unit":
        success = run_unit_tests()
    elif args.type == "integration":
        success = run_integration_tests()
    elif args.type == "performance":
        success = run_performance_tests()
    elif args.type == "isaac":
        success = run_isaac_sim_tests()
    elif args.type == "all":
        success = run_all_tests()
    else:
        print(f"无效的测试类型: {args.type}")
        return
    
    # 生成报告（如果请求）
    if args.report:
        report_success = generate_test_report()
        if not report_success:
            print("测试报告生成失败")
    
    elapsed_time = time.time() - start_time
    print(f"\\n测试总耗时: {elapsed_time:.2f} 秒")
    
    if success:
        print("所有测试完成")
        sys.exit(0)
    else:
        print("部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()


