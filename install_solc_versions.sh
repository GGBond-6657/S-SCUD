#!/bin/bash
# 安装常用的 Solidity 编译器版本

echo "=================================="
echo "安装 Solidity 编译器版本管理工具"
echo "=================================="

# 检查 solc-select 是否已安装
if ! command -v solc-select &> /dev/null; then
    echo "📦 安装 solc-select..."
    pip install solc-select
else
    echo "✅ solc-select 已安装"
fi

echo ""
echo "=================================="
echo "安装常用 Solidity 编译器版本"
echo "=================================="

# 定义要安装的版本
versions=(
    "0.4.25"
    "0.5.0"
    "0.5.17"
    "0.6.12"
    "0.8.0"
    "0.8.20"
)

# 安装每个版本
for version in "${versions[@]}"; do
    echo ""
    echo "📥 安装 Solidity $version..."
    solc-select install $version
    
    if [ $? -eq 0 ]; then
        echo "✅ Solidity $version 安装成功"
    else
        echo "❌ Solidity $version 安装失败"
    fi
done

echo ""
echo "=================================="
echo "查看已安装版本"
echo "=================================="
solc-select versions

echo ""
echo "=================================="
echo "设置默认版本为 0.5.0"
echo "=================================="
solc-select use 0.5.0

echo ""
echo "=================================="
echo "验证当前版本"
echo "=================================="
solc --version

echo ""
echo "✅ 安装完成！"
echo ""
echo "现在可以运行: python my_run.py"
