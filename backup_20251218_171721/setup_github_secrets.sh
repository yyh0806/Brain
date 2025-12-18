#!/bin/bash

echo "=== GitHub Secrets 自动配置脚本 ==="
echo ""
echo "此脚本将帮助您配置GitHub仓库的secrets"
echo ""

# 检查是否已登录GitHub CLI
if ! gh auth status >/dev/null 2>&1; then
    echo "❌ 需要先登录GitHub CLI"
    echo "请运行: gh auth login"
    echo "选择 'GitHub.com' 并按提示完成认证"
    exit 1
fi

echo "✅ GitHub CLI已登录"
echo ""

# 获取仓库信息
REPO_INFO=$(git remote get-url origin 2>/dev/null)
if [ -z "$REPO_INFO" ]; then
    echo "❌ 无法获取仓库信息，请确保在git仓库中运行此脚本"
    exit 1
fi

echo "当前仓库: $REPO_INFO"
echo ""

# 添加secrets
echo "正在添加GitHub Secrets..."

echo "添加 ANTHROPIC_API_KEY..."
if echo "060aa87524a7487ca9bc0357e3b1a5c8.6yfQD64rxI98ljb5" | gh secret set ANTHROPIC_API_KEY; then
    echo "✅ ANTHROPIC_API_KEY 添加成功"
else
    echo "❌ ANTHROPIC_API_KEY 添加失败"
fi

echo ""
echo "添加 ANTHROPIC_BASE_URL..."
if echo "https://open.bigmodel.cn/api/anthropic" | gh secret set ANTHROPIC_BASE_URL; then
    echo "✅ ANTHROPIC_BASE_URL 添加成功"
else
    echo "❌ ANTHROPIC_BASE_URL 添加失败"
fi

echo ""
echo "=== 配置完成 ==="
echo "您现在可以运行GitHub Actions来测试Claude Code集成"