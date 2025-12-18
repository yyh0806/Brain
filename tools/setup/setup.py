"""
Brain 系统安装配置
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# 读取requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="brain-autonomous-system",
    version="1.0.0",
    description="无人系统任务分解大脑 - 通过LLM将自然语言转换为机器人可执行操作",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Brain Team",
    author_email="",
    url="https://github.com/your-repo/brain",
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="autonomous systems, robotics, LLM, task planning, drone, UGV, USV",
    include_package_data=True,
    zip_safe=False,
)


