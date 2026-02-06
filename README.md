# S-SCUD: 智能合约审计工具

S-SCUD 是一个基于 LLM（大语言模型）的智能合约审计和分析平台，集成了 ChatDev 框架和多种 AI 模型，用于自动化分析和审计以太坊智能合约。

## 项目概述

本项目使用先进的 AI 技术（包括 OpenAI、Google Gemini、Anthropic Claude 等）来：
- 自动审计智能合约代码
- 检测潜在的安全漏洞
- 生成审计报告和建议
- 支持 CVE 漏洞检测和分析
- 执行合约静态分析

## 项目结构

```
S-SCUD/
├── camel/                      # CAMEL AI 框架模块
│   ├── agents/                 # 智能代理实现
│   │   ├── base.py            # 基础代理
│   │   ├── chat_agent.py       # 聊天代理
│   │   ├── task_agent.py       # 任务代理
│   │   └── tool_agents/        # 工具代理
│   ├── messages/               # 消息处理
│   ├── prompts/                # 提示模板
│   └── model_backend.py        # 模型后端支持
│
├── chatdev/                    # ChatDev 框架核心
│   ├── chat_chain.py          # 聊天链管理
│   ├── chat_env.py            # 环境配置
│   ├── phase.py               # 执行阶段
│   └── tools/                 # 集成工具
│
├── evaluation/                 # 评估和基准测试
│   └── benchmark/
│       └── contracts/          # 智能合约样本库
│           ├── cve/           # CVE 漏洞合约
│           ├── token/         # Token 合约
│           ├── access/        # 访问控制合约
│           └── ...
│
├── scripts/                    # 实用脚本
│   ├── vuln_detect_ckd.py     # 漏洞检测脚本
│   ├── prompt_builder.py      # 提示构建器
│   └── hybrid_detection_pipeline.py  # 混合检测管道
│
├── CompanyConfig/             # 配置文件
│   └── SmartContractCKD/      # 智能合约审计配置
│
├── WareHouse/                 # 审计输出目录
├── run.py                     # 主启动脚本
├── my_run.py                  # 批量审计脚本
├── requirements.txt           # 依赖包列表
└── README.md                  # 本文件
```

## 核心功能

### 1. 智能合约审计
- 支持 Solidity 合约分析
- 使用多种 AI 模型进行多角度审计
- 生成详细的审计报告

### 2. 漏洞检测
- 自动检测已知 CVE 漏洞
- 支持多种漏洞类型：
  - 重入漏洞
  - 整数溢出/下溢
  - 不安全委托调用
  - 时间依赖问题
  - 随机数问题
  - 等等

### 3. 多模型支持
支持以下 AI 模型：
- **OpenAI**: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o Mini
- **Google**: GLM-4.5 Air, Gemini 3 Pro, Gemini 3 Pro Thinking, Gemini Flash Search
- **Anthropic**: Claude Haiku 4.5, Claude Sonnet 4.5

## 安装和配置

### 1. 环境要求
- Python 3.8+
- pip 包管理工具

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥
设置环境变量以配置你的 AI 模型 API 密钥：
```bash
export OPENAI_API_KEY="your-openai-api-key"
# 其他模型配置类似...
```

## 使用方法

### 方式一：单个合约审计
```bash
python run.py \
  --config SmartContractCKD \
  --org YourOrganization \
  --name YourContractName \
  --task "Your contract code or description" \
  --model GPT_4_O_MINI
```

### 方式二：指定 Solidity 文件
```bash
python run.py \
  --sol /path/to/contract.sol \
  --config SmartContractCKD \
  --org YourOrganization \
  --name YourContractName \
  --model GPT_4_O_MINI
```

### 方式三：批量审计合约
```bash
python my_run.py
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件名（位于 CompanyConfig 下） | Default |
| `--org` | 组织名称 | DefaultOrganization |
| `--name` | 项目名称 | Gomoku |
| `--task` | 审计任务描述或合约代码 | Develop a basic Gomoku game. |
| `--model` | 使用的 AI 模型 | GPT_4_O_MINI |
| `--path` | 增量模式下的文件目录 | 空 |
| `--sol` | Solidity 合约文件路径 | 空 |

## 配置文件

### ChatChainConfig.json
定义聊天链的整体配置，包括模型参数、温度、最大 token 数等。

### PhaseConfig.json
定义审计的各个阶段（如需求分析、代码审查、安全检查等）。

### RoleConfig.json
定义参与审计的各种角色（如安全审计员、代码审查员等）。

## 输出

审计结果会保存在 `WareHouse/` 目录下，按照以下结构组织：
```
WareHouse/
└── ProjectName_Organization_Timestamp/
    ├── audit_report.md        # 审计报告
    ├── vulnerabilities.json   # 漏洞列表
    └── ...
```

## 依赖包

主要依赖包括：
- `openai>=1.3.3` - OpenAI API
- `Flask>=2.3.2` - Web 框架
- `beautifulsoup4>=4.12.2` - HTML/XML 解析
- `pyyaml>=6.0` - YAML 配置
- `requests>=2.31.0` - HTTP 请求
- 更多详见 `requirements.txt`

**注意**: 使用本工具前，请确保已获得必要的 API 访问权限，并遵守相关服务的使用条款。
