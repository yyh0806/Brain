# Brain项目Git工作流程

本文档描述了如何使用Git Worktree方式管理Brain项目的各个层级模块，支持并行开发和独立维护。

## 初始设置

### 1. 克隆仓库并设置主分支

```bash
# 克隆主仓库
git clone <repository-url> Brain
cd Brain

# 确保在main分支
git checkout main

# 创建并切换到develop分支（用于日常开发）
git checkout -b develop
git push -u origin develop
```

### 2. 创建各层级的Worktree

```bash
# 在Brain目录下创建各层级的worktree
git worktree add ../brain-perception perception
git worktree add ../brain-cognitive cognitive
git worktree add ../brain-planning planning
git worktree add ../brain-execution execution
git worktree add ../brain-communication communication
git worktree add ../brain-models models
git worktree add ../brain-core core
git worktree add ../brain-platforms platforms
git worktree add ../brain-recovery recovery
git worktree add ../brain-state state
git worktree add ../brain-utils utils
git worktree add ../brain-visualization visualization
```

## 分支策略

### 主分支

- `main`: 稳定发布分支，包含所有层级
- `develop`: 日常开发分支，定期合并各层级的开发内容

### 层级分支

- `perception-dev`: 感知层开发分支
- `cognitive-dev`: 认知层开发分支
- `planning-dev`: 规划层开发分支
- `execution-dev`: 执行层开发分支
- `communication-dev`: 通信层开发分支
- `models-dev`: 模型层开发分支
- `core-dev`: 核心控制器开发分支
- `platforms-dev`: 平台支持开发分支
- `recovery-dev`: 错误恢复开发分支
- `state-dev`: 状态管理开发分支
- `utils-dev`: 工具类开发分支
- `visualization-dev`: 可视化开发分支

### 功能分支

命名规范: `feature/<功能名>-<层级>`

示例:
- `feature/object-detection-perception`: 感知层物体检测功能
- `feature/dialogue-management-cognitive`: 认知层对话管理功能
- `feature/path-planning-planning`: 规划层路径规划功能
- `feature/operation-execution`: 执行层操作功能
- `feature/ros2-interface-communication`: 通信层ROS2接口功能
- `feature/llm-integration-models`: 模型层LLM集成功能

## 开发工作流

### 1. 开始新功能开发

```bash
# 1. 从develop分支创建功能分支
git checkout develop
git pull origin develop
git checkout -b feature/new-functionality-layer

# 2. 进入对应的worktree进行开发
cd ../brain-<layer>
# 例如，开发感知层功能
cd ../brain-perception

# 3. 进行代码修改...
# 编辑文件，添加功能...

# 4. 提交代码
git add .
git commit -m "feat: add new functionality to <layer> module"
```

### 2. 开发过程中的同步

```bash
# 定期从develop分支同步最新代码
git checkout develop
git pull origin develop

# 合并到功能分支
git checkout feature/new-functionality-layer
git merge develop

# 解决可能的冲突...
git push origin feature/new-functionality-layer
```

### 3. 功能开发完成

```bash
# 1. 确保代码已提交
git status
git push origin feature/new-functionality-layer

# 2. 创建Pull Request到develop分支
# 在GitHub/GitLab上创建PR，请求代码审查
```

### 4. 代码审查和合并

```bash
# 1. 代码审查通过后，合并到层级开发分支
git checkout perception-dev  # 或其他层级分支
git merge feature/new-functionality-layer
git push origin perception-dev

# 2. 定期将层级开发分支合并到develop分支
git checkout develop
git merge perception-dev
git merge cognitive-dev
git merge planning-dev
git merge execution-dev
git merge communication-dev
git merge models-dev
git merge core-dev
git merge platforms-dev
git merge recovery-dev
git merge state-dev
git merge utils-dev
git merge visualization-dev
git push origin develop
```

### 5. 发布流程

```bash
# 1. 从develop分支创建release分支
git checkout develop
git pull origin develop
git checkout -b release/v1.0.0

# 2. 更新版本号和更新日志
# 编辑相关文件...

# 3. 合并到main分支
git checkout main
git merge release/v1.0.0 --no-ff
git tag v1.0.0
git push origin main --tags

# 4. 合并回develop分支
git checkout develop
git merge main
git push origin develop
```

## Worktree管理

### 查看所有Worktree

```bash
cd Brain
git worktree list
```

### 删除Worktree

```bash
cd Brain
git worktree remove ../brain-<layer>
```

### 同步所有Worktree

```bash
# 在主仓库中更新所有worktree
cd Brain
git fetch origin

# 更新每个worktree
cd ../brain-perception && git pull origin perception-dev
cd ../brain-cognitive && git pull origin cognitive-dev
cd ../brain-planning && git pull origin planning-dev
cd ../brain-execution && git pull origin execution-dev
cd ../brain-communication && git pull origin communication-dev
cd ../brain-models && git pull origin models-dev
cd ../brain-core && git pull origin core-dev
cd ../brain-platforms && git pull origin platforms-dev
cd ../brain-recovery && git pull origin recovery-dev
cd ../brain-state && git pull origin state-dev
cd ../brain-utils && git pull origin utils-dev
cd ../brain-visualization && git pull origin visualization-dev
```

## 跨层级依赖管理

### 1. 接口变更

当需要修改接口时，应遵循以下流程：

1. 在对应的层级分支中实现新接口
2. 创建功能分支: `feature/interface-change-<affected-layer>`
3. 实现接口变更
4. 更新依赖此接口的其他层级
5. 提交并创建PR
6. 通知所有相关开发人员接口变更

### 2. 依赖更新

当需要更新依赖时：

1. 在requirements.txt中更新依赖版本
2. 在各层级中测试兼容性
3. 创建功能分支: `feature/dependency-update-<affected-layers>`
4. 提交并创建PR

## 测试策略

### 1. 单元测试

```bash
# 在对应层级的worktree中运行
cd ../brain-<layer>
python -m pytest tests/<layer>/unit/

# 运行特定测试
python -m pytest tests/<layer>/unit/test_specific_module.py
```

### 2. 集成测试

```bash
# 在主仓库中运行跨层级集成测试
cd Brain
python -m pytest tests/integration/
```

### 3. 系统测试

```bash
# 在主仓库中运行完整系统测试
cd Brain
python -m pytest tests/system/
```

## 常见问题解决

### 1. Worktree冲突

```bash
# 如果worktree出现冲突，可以重新创建
cd Brain
git worktree remove ../brain-<layer>
git worktree add ../brain-<layer> <branch>
```

### 2. 分支合并冲突

```bash
# 解决合并冲突
git checkout feature/branch-name
git merge develop
# 手动解决冲突文件...
git add .
git commit -m "resolve merge conflicts"
git push origin feature/branch-name
```

### 3. 同步问题

```bash
# 如果某个worktree无法同步，可以重置
cd ../brain-<layer>
git reset --hard origin/<branch>
```

## 最佳实践

1. **保持分支专注**: 每个功能分支只关注一个特定功能
2. **频繁同步**: 定期从develop分支同步最新代码
3. **小步提交**: 保持提交小而频繁，便于代码审查
4. **清晰提交信息**: 使用规范的提交信息格式
5. **及时沟通**: 跨层级变更时及时沟通协调
6. **测试先行**: 提交前确保相关测试通过
7. **文档同步**: 接口变更时同步更新文档

## 提交信息规范

```
feat: 新功能
fix: 修复bug
docs: 文档更新
style: 代码格式调整
refactor: 代码重构
test: 测试相关
chore: 构建过程或辅助工具的变动

示例:
feat(perception): add object detection module
fix(cognitive): resolve dialogue manager memory leak
docs(planning): update navigation planner documentation
```

## 发布检查清单

- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] 版本号已更新
- [ ] 变更日志已编写
- [ ] 性能测试通过
- [ ] 安全审查完成
- [ ] 跨层级兼容性确认
