# 🚀 Brain项目开发规范

## 📋 分支策略

### 分支命名规范
- `master` - 主分支，用于生产环境
- `develop` - 开发分支，集成最新功能
- `feature/*` - 新功能开发分支
- `bugfix/*` - Bug修复分支
- `hotfix/*` - 紧急修复分支
- `release/*` - 发布准备分支

### 分支保护规则
- ✅ **master分支**:
  - 禁止直接推送
  - 必须通过Pull Request合并
  - 需要至少1个审核人批准
  - 必须通过所有CI/CD检查

- ✅ **develop分支**:
  - 建议通过Pull Request合并
  - 需要通过基础CI检查

## 🔄 工作流程

### 1. 功能开发流程
```bash
# 1. 从develop创建功能分支
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# 2. 开发和提交
git add .
git commit -m "feat: 添加新功能描述"

# 3. 推送分支
git push origin feature/your-feature-name

# 4. 创建Pull Request
# - 目标分支: develop
# - 使用PR模板填写详细信息

# 5. 合并后清理
git checkout develop
git pull origin develop
git branch -d feature/your-feature-name
```

### 2. Bug修复流程
```bash
# 1. 从master创建bugfix分支
git checkout master
git pull origin master
git checkout -b bugfix/issue-number-description

# 2. 修复和测试
# ... 修复代码 ...
git commit -m "fix: 修复问题描述 (fixes #issue-number)"

# 3. 创建Pull Request到master
# - 需要紧急修复可直接到master
# - 一般修复先到develop再合并到master
```

### 3. 紧急修复流程
```bash
# 1. 从master创建hotfix分支
git checkout master
git pull origin master
git checkout -b hotfix/critical-fix-description

# 2. 快速修复和部署
git commit -m "hotfix: 紧急修复描述"
git push origin hotfix/critical-fix

# 3. 创建Pull Request到master
# - 标记为紧急修复
# - 优先审核和合并
```

## 📝 提交信息规范

### 提交类型
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式调整（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建工具、依赖更新等

### 提交格式
```
<类型>(<范围>): <描述>

[可选的详细描述]

[可选的相关Issue: #123]
```

示例：
```
feat(perception): 添加新的传感器数据处理模块

- 实现了数据预处理流程
- 添加了错误处理机制
- 更新了相关文档

Related: #45
```

## 🔍 代码审核规范

### 审核人职责
1. **代码质量检查**
   - 代码逻辑是否正确
   - 是否符合编码规范
   - 是否有潜在的性能问题

2. **安全性检查**
   - 是否有安全漏洞
   - 敏感信息是否正确处理
   - 输入验证是否充分

3. **测试覆盖**
   - 是否有足够的测试
   - 测试用例是否合理
   - 边界条件是否考虑

### 审核检查清单
- [ ] 代码逻辑清晰易懂
- [ ] 变量和函数命名合理
- [ ] 有必要的注释
- [ ] 无明显性能问题
- [ ] 考虑了错误处理
- [ ] 测试覆盖充分
- [ ] 文档已更新
- [ ] 符合项目编码规范

## 🧪 测试要求

### 测试类型
1. **单元测试**
   - 每个模块的核心功能
   - 边界条件测试
   - 错误处理测试

2. **集成测试**
   - 模块间交互测试
   - API接口测试
   - 数据流测试

3. **系统测试**
   - 端到端功能测试
   - 性能测试
   - 稳定性测试

### 测试覆盖率要求
- 新代码测试覆盖率 > 80%
- 关键模块测试覆盖率 > 90%

## 📚 文档要求

### 必需文档
1. **README.md** - 项目概述和快速开始
2. **API文档** - 接口说明和示例
3. **架构文档** - 系统设计说明
4. **部署文档** - 部署和配置说明

### 文档更新时机
- 新功能开发时
- API变更时
- 架构调整时
- 部署流程变更时

## 🔧 开发环境配置

### 本地开发
```bash
# 1. 克隆仓库
git clone https://github.com/yyh0806/Brain.git
cd Brain

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装开发工具
pip install -r requirements-dev.txt

# 5. 运行测试
pytest

# 6. 代码格式化
black brain/
isort brain/
```

### 代码质量工具
- **black**: 代码格式化
- **isort**: import排序
- **flake8**: 代码检查
- **mypy**: 类型检查
- **pytest**: 测试框架

## 🚀 CI/CD流程

### 自动化检查
1. **代码质量检查**
   - 代码格式检查
   - 代码风格检查
   - 类型检查

2. **安全检查**
   - 敏感信息扫描
   - 依赖安全检查
   - 代码安全分析

3. **测试执行**
   - 单元测试
   - 集成测试
   - 覆盖率检查

### 部署流程
1. **开发环境** → 自动部署
2. **测试环境** → 手动触发
3. **生产环境** → 审批后部署

## 📊 监控和日志

### 日志规范
- 使用统一的日志格式
- 包含时间戳、级别、模块、消息
- 关键操作添加详细日志

### 监控指标
- 系统性能指标
- 业务指标
- 错误率统计

## 🔐 安全规范

### 敏感信息处理
- API密钥使用环境变量
- 密码和token加密存储
- 敏感配置文件不提交到仓库

### 代码安全
- 输入验证和过滤
- SQL注入防护
- XSS攻击防护

## 📞 联系方式

### 技术支持
- 项目维护者: [维护者信息]
- 技术讨论: [讨论渠道]
- Bug报告: [GitHub Issues]

### 文档
- 项目文档: [文档链接]
- API文档: [API文档链接]
- 架构设计: [设计文档链接]

---

## 📋 快速检查清单

在提交代码前，请确认：
- [ ] 代码已通过本地测试
- [ ] 提交信息符合规范
- [ ] 代码已格式化
- [ ] 文档已更新
- [ ] 没有敏感信息泄露
- [ ] 测试覆盖率达标
- [ ] 已创建Pull Request
- [ ] 已填写PR模板
- [ ] 已指定审核人

---
*最后更新: 2025-12-18*