# Brain项目开发示例

本文档提供了使用Brain项目开发工具的示例，帮助开发人员快速上手。

## 快速开始

### 1. 初始化开发环境

```bash
# 在Brain项目根目录下运行
./setup_dev_env.sh
```

这将创建以下结构：
```
Brain/
├── brain/                    # 主代码仓库
├── brain-worktrees/           # Worktree目录
│   ├── brain-perception/     # 感知层Worktree
│   ├── brain-cognitive/     # 认知层Worktree
│   ├── brain-planning/      # 规划层Worktree
│   ├── brain-execution/      # 执行层Worktree
│   ├── brain-communication/  # 通信层Worktree
│   └── brain-models/        # 模型层Worktree
└── scripts/                  # 便捷脚本
    ├── goto_layer.sh
    ├── sync_all.sh
    ├── commit_all.sh
    └── merge_to_develop.sh
```

### 2. 切换到指定层级开发环境

```bash
# 切换到感知层开发环境
./scripts/goto_layer.sh perception

# 切换到认知层开发环境
./scripts/goto_layer.sh cognitive

# 切换到规划层开发环境
./scripts/goto_layer.sh planning
```

## 开发工作流示例

### 示例1: 开发感知层新功能

```bash
# 1. 切换到感知层环境
./scripts/goto_layer.sh perception

# 2. 创建功能分支
git checkout -b feature/new-sensor-perception

# 3. 进行开发...
# 编辑相关文件...

# 4. 提交代码
git add .
git commit -m "feat(perception): add new sensor support"

# 5. 推送到远程
git push origin feature/new-sensor-perception

# 6. 创建PR到perception-dev分支
# 在GitHub/GitLab上创建Pull Request
```

### 示例2: 跨层级功能开发

假设需要开发一个新功能，涉及感知层和认知层：

```bash
# 1. 在主仓库创建功能分支
git checkout develop
git checkout -b feature/cross-layer-functionality

# 2. 切换到感知层环境，开发感知部分
./scripts/goto_layer.sh perception
git checkout feature/cross-layer-functionality

# 开发感知部分代码...
git add .
git commit -m "feat(perception): add perception part for cross-layer functionality"
git push origin feature/cross-layer-functionality

# 3. 切换到认知层环境，开发认知部分
./scripts/goto_layer.sh cognitive
git checkout feature/cross-layer-functionality

# 开发认知部分代码...
git add .
git commit -m "feat(cognitive): add cognitive part for cross-layer functionality"
git push origin feature/cross-layer-functionality

# 4. 创建PR到develop分支
# 在GitHub/GitLab上创建Pull Request
```

### 示例3: 批量同步和提交

```bash
# 1. 同步所有层级
./scripts/sync_all.sh

# 2. 批量提交所有层级的更改
./scripts/commit_all.sh "feat: implement batch processing"

# 3. 合并所有层级到develop分支
./scripts/merge_to_develop.sh
```

## 测试示例

### 运行层级测试

```bash
# 测试所有层级的导入和基本功能
python3 test_layers.py
```

预期输出：
```
Brain项目层级测试
============================================================
测试 perception 层级
============================================================
  ✓ environment
  ✓ object_detector
  ✓ sensors
  ✓ mapping
  ✓ vlm

测试 cognitive 层级
============================================================
  ✓ world_model
  ✓ dialogue
  ✓ reasoning
  ✓ monitoring

测试 planning 层级
============================================================
  ✓ task
  ✓ navigation
  ✓ behavior

测试 execution 层级
============================================================
  ✓ executor
  ✓ operations

测试 communication 层级
============================================================
  ✓ robot_interface
  ✓ ros2_interface
  ✓ control_adapter
  ✓ message_types

测试 models 层级
============================================================
  ✓ llm_interface
  ✓ task_parser
  ✓ prompt_templates
  ✓ ollama_client
  ✓ cot_prompts

测试跨层级导入
============================================================
  ✓ core -> perception
  ✓ core -> cognitive
  ✓ core -> planning
  ✓ core -> execution
  ✓ core -> communication
  ✓ core -> models

测试Brain核心初始化
============================================================
2025-12-16 17:00:07.441 | WARNING  | brain.cognitive.world_model.world_model:<module>:29 - VLM perception not available
Brain导入成功

测试结果汇总
============================================================
总测试数: 24
成功测试数: 24
成功率: 100.0%

所有测试通过！
```

## 常见问题解决

### 1. Worktree同步问题

```bash
# 如果某个worktree无法同步，可以重置
cd ../brain-perception
git reset --hard origin/perception-dev

# 或者重新创建worktree
cd Brain
git worktree remove ../brain-perception
git worktree add ../brain-perception perception
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

### 3. 依赖更新问题

```bash
# 更新依赖后，在所有层级中测试兼容性
./scripts/sync_all.sh
python3 test_layers.py

# 如果有测试失败，逐个层级排查
./scripts/goto_layer.sh perception
python3 -c "from brain.perception import environment; print('OK')"
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
