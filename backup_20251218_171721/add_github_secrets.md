# GitHub Secrets 配置指南

## 自动添加脚本

如果您有GitHub CLI并且已登录，可以使用以下命令：

```bash
# 添加 ANTHROPIC_API_KEY
gh secret set ANTHROPIC_API_KEY --body "060aa87524a7487ca9bc0357e3b1a5c8.6yfQD64rxI98ljb5"

# 添加 ANTHROPIC_BASE_URL
gh secret set ANTHROPIC_BASE_URL --body "https://open.bigmodel.cn/api/anthropic"
```

## 手动添加步骤

### 步骤1: 进入仓库Settings
- 点击仓库顶部的 "Settings" 标签

### 步骤2: 找到Secrets菜单
在左侧菜单中寻找以下选项之一：
- "Secrets and variables" → "Actions" (新版)
- "Secrets" (旧版)

### 步骤3: 添加Secret
点击 "New repository secret" 或 "Add a new secret" 按钮

### 步骤4: 填写信息
**Secret 1:**
- Name: `ANTHROPIC_API_KEY`
- Value: `060aa87524a7487ca9bc0357e3b1a5c8.6yfQD64rxI98ljb5`

**Secret 2:**
- Name: `ANTHROPIC_BASE_URL`
- Value: `https://open.bigmodel.cn/api/anthropic`

## 验证配置

添加完成后，您可以：
1. 运行测试workflow
2. 检查是否显示为 configured secrets