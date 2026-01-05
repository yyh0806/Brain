#!/usr/bin/env python3
"""
感知层测试运行脚本

使用方法:
  python run_tests.py              # 运行所有测试
  python run_tests.py unit         # 只运行单元测试
  python run_tests.py functional   # 只运行功能测试
  python run_tests.py integration  # 只运行集成测试
  python run_tests.py -v           # 详细模式
  python run_tests.py -k "test_"   # 运行匹配的测试
"""

import sys
import argparse
import pytest


def main():
    parser = argparse.ArgumentParser(description="运行感知层测试")
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=["unit", "functional", "integration"],
        help="测试类型: unit, functional, integration (默认: all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "-k",
        dest="keyword",
        help="只运行匹配关键字的测试"
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="生成代码覆盖率报告"
    )
    parser.add_argument(
        "--cov-report",
        default="term-missing",
        help="覆盖率报告格式 (默认: term-missing)"
    )

    args = parser.parse_args()

    # 构建pytest参数
    pytest_args = []

    # 测试目录
    if args.test_type:
        test_path = f"tests/perception/{args.test_type}"
        pytest_args.append(test_path)
    else:
        pytest_args.append("tests/perception")

    # 详细模式
    if args.verbose:
        pytest_args.append("-v")

    # 关键字过滤
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # 代码覆盖率
    if args.cov:
        pytest_args.extend([
            "--cov=brain/perception",
            f"--cov-report={args.cov_report}"
        ])

    # 添加颜色输出
    pytest_args.append("--color=yes")

    # 运行测试
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
