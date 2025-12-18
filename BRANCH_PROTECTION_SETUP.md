# 🔒 分支保护设置指南

## 🎯 问题解决

当前GitHub显示"Your master branch isn't protected"，这是因为我们需要手动在GitHub网页端设置分支保护规则。

## 🛠️ 设置方法

### 方法1: GitHub网页端设置（推荐）

1. **访问仓库设置**
   - 打开 https://github.com/yyh0806/Brain
   - 点击 **Settings** 标签

2. **进入分支设置**
   - 在左侧菜单中点击 **Branches**
   - 找到 **Branch protection rules** 部分
   - 点击 **Add rule** 按钮

3. **配置保护规则**
   ```
   Branch name pattern: master
   ☑️ Require a pull request before merging
   ☑️ Require approvals: 1
   ☑️ Dismiss stale PR approvals when new commits are pushed
   ☑️ Require review from CODEOWNERS (可选)

   ☑️ Require status checks to pass before merging
   ☑️ Require branches to be up to date before merging
   ☑️ Status checks that are required: CI
   ☑️ Code quality checks
   ☑️ Security scan
   ```

4. **保存设置**
   - 点击 **Create** 或 **Save changes**
   - 设置立即生效

### 方法2: GitHub API设置（高级用户）

由于GitHub API的格式要求比较严格，这里是正确的API调用格式：

```bash
curl -X PUT \
  -H "Authorization: token YOUR_GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/yyh0806/Brain/branches/master/protection \
  -d '{
    "required_status_checks": {
      "strict": true,
      "contexts": ["CI", "code-quality", "security-scan", "test"]
    },
    "enforce_admins": true,
    "required_pull_request_reviews": {
      "required_approving_review_count": 1,
      "dismiss_stale_reviews": true,
      "require_code_owner_reviews": false
    },
    "restrictions": null
  }'
```

### ✅ 状态检查已可用且完全修复！

通过PR #2测试，以下状态检查现在已经在您的仓库中可用并全部通过：

- ✅ **CI** - 基础持续集成检查 (SUCCESS)
- ✅ **code-quality** - 代码质量检查 (SUCCESS)
- ✅ **security-scan** - 安全扫描 (SUCCESS)
- ✅ **test** - 测试运行 (SUCCESS)

**🎉 重要更新**: CI/CD依赖问题已完全修复！
- 移除了不存在的依赖包
- 创建了轻量级CI环境
- 所有状态检查现在正常运行

这些状态检查现在应该出现在分支保护设置的"Status checks that are required"下拉列表中！

## ✅ 验证设置

设置完成后，您应该看到：

1. **分支保护图标** 🛡️
   - master分支旁边出现盾牌图标
   - 页面显示"Branch protection: 1 required status check"

2. **推送限制** 🚫
   - 无法直接推送到master分支
   - 必须通过Pull Request

3. **PR要求** 📋
   - 必须创建PR才能合并
   - 需要审核人批准
   - 必须通过CI检查

## 🔄 测试流程

设置后测试一下新的工作流程：

### 1. 尝试直接推送（应该失败）
```bash
echo "test change" >> test.txt
git add test.txt
git commit -m "test direct push"
git push origin master
# 应该显示错误，说明分支保护生效
```

### 2. 正确的PR流程
```bash
# 创建功能分支
git checkout -b feature/test-protection
git add test.txt
git commit -m "feat: 测试分支保护"
git push origin feature/test-protection

# 创建PR并测试审核流程
```

## ⚙️ 可选的高级设置

### 1. CODEOWNERS配置
创建 `.github/CODEOWNERS` 文件：
```
# 全局设置
* @yyh0806

# 特定目录
brain/ @team-lead
docs/ @documentation-team
```

### 2. 多级审核要求
- 重要文件需要2个审核人
- 安全相关文件需要安全专家审核
- 文档变更需要文档团队审核

### 3. 自动化检查
- 代码覆盖率 > 80%
- 性能测试通过
- 文档完整性检查

## 🎯 预期效果

设置完成后，您的开发流程将变成：

1. **🚫 禁止直接推送**
   - master分支受到保护
   - 所有变更必须通过PR

2. **👥 强制代码审核**
   - 至少需要1个审核人
   - 审核人必须是项目成员

3. **✅ 自动化检查**
   - CI/CD必须通过
   - 代码质量检查
   - 安全扫描

4. **📝 标准化流程**
   - 使用PR模板
   - 详细的提交信息
   - 完整的文档更新

---

*建议现在就按照方法1的步骤在GitHub网页端设置分支保护！*