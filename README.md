# sensor-input-dev 开发环境

## 分支信息
- 工作树: sensor-input-dev
- 分支: sensor-input-dev
- 基础分支: master

## 模块职责
传感器数据输入模块开发
- 点云传感器接口
- 视觉传感器接口
- 多传感器同步
- 数据格式标准化

## 开发指南
1. 基于主分支的最新代码进行开发
2. 遵循项目的编码规范
3. 及时提交代码并编写测试
4. 定期同步主分支的更新

## 测试命令
```bash
# 运行单元测试
python -m pytest tests/unit/

# 运行集成测试
python -m pytest tests/integration/

# 生成测试覆盖率报告
python -m pytest --cov=brain tests/
```

## 提交规范
提交信息格式:
`模块: 简短描述`

例如:
`sensor-input: 激光雷达数据接口实现`

---
创建时间: 2025年 12月 17日 星期三 09:39:00 CST
