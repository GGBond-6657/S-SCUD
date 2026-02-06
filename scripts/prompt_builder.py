#!/usr/bin/env python3
"""
LLM Prompt Constructor - 构造结构化漏洞检测提示

基于蒸馏后的路径上下文，生成专业的安全分析提示
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VulnerabilityPrompt:
    """漏洞检测提示"""
    contract: str
    function: str
    prompt: str
    metadata: Dict


class PromptBuilder:
    """提示构造器"""
    
    def __init__(self, distilled_file: str, source_file: Optional[str] = None):
        """
        初始化提示构造器
        
        Args:
            distilled_file: 蒸馏结果 JSON 文件
            source_file: 原始 Solidity 源文件（可选，用于添加完整代码）
        """
        self.distilled_file = Path(distilled_file)
        if not self.distilled_file.exists():
            raise FileNotFoundError(f"蒸馏文件不存在: {distilled_file}")
        
        # 加载蒸馏结果
        with open(self.distilled_file, 'r', encoding='utf-8') as f:
            self.contexts = json.load(f)
        
        # 加载源代码（可选）
        self.source_code = None
        if source_file:
            source_path = Path(source_file)
            if source_path.exists():
                self.source_code = source_path.read_text(encoding='utf-8')
        
        logger.info(f"[Prompt] 加载了 {len(self.contexts)} 个函数的蒸馏上下文")
    
    def build_all(self, language: str = 'zh') -> List[VulnerabilityPrompt]:
        """
        为所有函数构建提示
        
        Args:
            language: 'zh' (中文) 或 'en' (英文)
        
        Returns:
            提示列表
        """
        prompts = []
        
        for ctx in self.contexts:
            prompt = self.build_single(ctx, language)
            prompts.append(prompt)
        
        return prompts
    
    def build_single(self, context: Dict, language: str = 'zh') -> VulnerabilityPrompt:
        """
        为单个函数构建提示
        
        Args:
            context: 蒸馏上下文（DistilledContext 的字典形式）
            language: 'zh' 或 'en'
        
        Returns:
            VulnerabilityPrompt 对象
        """
        if language == 'zh':
            return self._build_zh(context)
        else:
            return self._build_en(context)
    
    def _build_zh(self, ctx: Dict) -> VulnerabilityPrompt:
        """构建中文提示"""
        
        # ===== 1. 角色定义 =====
        role = "你是一位智能合约安全专家，擅长识别 Solidity 代码中的漏洞。"
        
        # ===== 2. 任务说明 =====
        instruction = f"""
## 任务说明
下面是从智能合约 `{ctx['contract']}` 的函数 `{ctx['function_signature']}` 中提取的**关键执行路径切片**。

这些路径会执行**状态修改或外部调用等操作**。请注意：
- ⚠️ **并非所有状态修改都是漏洞**（正常的业务逻辑也需要修改状态）
- ✅ **重点关注守卫条件是否充分**（require/if/modifier 是否能防止攻击）

你的任务是：
1. **评估守卫条件的有效性**：
   - 是否检查了零地址？
   - 是否检查了权限（onlyOwner/msg.sender）？
   - 是否防止了重入攻击？
   - 是否检查了余额/溢出？

2. **判断是否存在可被绕过的检查或逻辑缺陷**

3. **回答：该函数是否存在可利用的漏洞？（是/否）**
   - 如果守卫条件充分且逻辑正确，应回答"否"
   - 如果存在可被利用的缺陷，应回答"是"

4. 如果存在漏洞，请：
   - 说明**漏洞类型**（重入、访问控制、整数溢出等）
   - 引用**具体的路径 ID**
   - 给出**最小修复建议**
"""
        
        # ===== 3. 上下文信息 =====
        context_section = self._build_context_section_zh(ctx)
        
        # ===== 4. 路径切片详情 =====
        paths_section = self._build_paths_section_zh(ctx)
        
        # ===== 5. 问题 =====
        question = """
## 安全评估

**请客观评估以下问题：**

1. **守卫条件评估**：
   - 是否检查了关键参数（地址非零、金额合法等）？
   - 是否有访问控制（modifier/require检查调用者）？
   - 是否防止了重入攻击（nonReentrant/Checks-Effects-Interactions）？

2. **该函数是否存在可利用的漏洞？（是/否）**
   - **如果守卫充分且逻辑正确**：回答"否"
   - **如果存在可被利用的缺陷**：回答"是"

3. **如果存在漏洞**：
   - 漏洞类型？（重入、访问控制缺陷、未检查返回值、整数溢出/下溢、逻辑错误等）
   - 受影响的路径 ID？
   - 攻击场景？
   - 修复建议？

**请以结构化格式回答：**
```
守卫条件评估: [充分/不充分/部分充分]
漏洞存在: [是/否]
漏洞类型: [类型名称，如无则填"无"]
受影响路径: [路径ID列表，如无则填"无"]
攻击场景: [简要说明，如无则填"无"]
修复建议: [具体建议，如无则填"无需修复"]
```
"""
        
        # ===== 组装完整提示 =====
        full_prompt = f"{role}\n\n{instruction}\n\n{context_section}\n\n{paths_section}\n\n{question}"
        
        return VulnerabilityPrompt(
            contract=ctx['contract'],
            function=ctx['function_signature'],
            prompt=full_prompt,
            metadata={
                'total_risk_score': ctx['total_risk_score'],
                'num_slices': len(ctx['path_slices']),
                'recommended_slices': ctx['recommended_slices']
            }
        )
    
    def _build_context_section_zh(self, ctx: Dict) -> str:
        """构建上下文部分（中文）"""
        lines = ["## 上下文信息\n"]
        
        # 合约基本信息
        lines.append(f"**合约名称**: `{ctx['contract']}`")
        lines.append(f"**函数签名**: `{ctx['function_signature']}`")
        lines.append(f"**风险评分**: {ctx['total_risk_score']:.1f}\n")
        
        # 状态变量定义
        if ctx['state_var_definitions']:
            lines.append("### 相关状态变量")
            lines.append("```solidity")
            for var_name, var_def in ctx['state_var_definitions'].items():
                lines.append(f"{var_def};")
            lines.append("```\n")
        
        # 依赖函数
        if ctx['dependent_function_code']:
            lines.append("### 依赖的内部函数")
            for func_name, func_info in ctx['dependent_function_code'].items():
                lines.append(f"\n**函数**: `{func_info['signature']}`")
                lines.append(f"- 可见性: {func_info['visibility']}")
                if func_info['modifiers']:
                    lines.append(f"- 修饰符: {', '.join(func_info['modifiers'])}")
                
                if func_info['expressions']:
                    lines.append("- 关键逻辑:")
                    lines.append("```solidity")
                    for expr in func_info['expressions'][:5]:  # 限制 5 行
                        lines.append(f"  {expr}")
                    lines.append("```")
        
        return "\n".join(lines)
    
    def _build_paths_section_zh(self, ctx: Dict) -> str:
        """构建路径切片部分（中文）"""
        lines = ["## 执行路径切片\n"]
        
        for i, path in enumerate(ctx['path_slices'], 1):
            lines.append(f"### 路径 {i}: `{path['slice_id']}`")
            lines.append(f"**风险评分**: {path['risk_score']:.1f}")
            lines.append(f"**Sink 类型**: {path['sink_type']}")
            
            # 风险因素
            if path['risk_factors']:
                lines.append("**风险因素**:")
                for factor in path['risk_factors']:
                    lines.append(f"- {factor}")
            
            # 守卫条件
            if path['guards']:
                lines.append("\n**守卫条件** (必须满足才能到达 Sink):")
                lines.append("```solidity")
                for guard in path['guards']:
                    lines.append(f"  {guard}")
                lines.append("```")
            else:
                lines.append("\n⚠️ **该路径没有守卫条件！**")
            
            # 状态变量访问
            if path['state_vars_read'] or path['state_vars_written']:
                lines.append("\n**状态变量访问**:")
                if path['state_vars_read']:
                    lines.append(f"- 读取: {', '.join(path['state_vars_read'])}")
                if path['state_vars_written']:
                    lines.append(f"- 写入: {', '.join(path['state_vars_written'])}")
            
            # 关键表达式
            if path['expressions']:
                lines.append("\n**执行流程** (从入口到 Sink):")
                lines.append("```solidity")
                for expr in path['expressions'][:10]:  # 限制 10 行
                    lines.append(f"  {expr}")
                if len(path['expressions']) > 10:
                    lines.append(f"  ... (省略 {len(path['expressions']) - 10} 行)")
                lines.append("```")
            
            lines.append("\n---\n")
        
        return "\n".join(lines)
    
    def _build_en(self, ctx: Dict) -> VulnerabilityPrompt:
        """构建英文提示（简化版）"""
        role = "You are a smart contract security expert specialized in identifying Solidity vulnerabilities."
        
        instruction = f"""
## Task
Analyze the following execution path slices from function `{ctx['function_signature']}` in contract `{ctx['contract']}`.

These paths lead to **sensitive operations** (e.g., transfers, state changes, external calls).

**Your task:**
1. Analyze the **constraints** (require/if statements) on each path
2. Identify any **bypassable checks** or **logic flaws**
3. Answer: **Does this function contain exploitable vulnerabilities? (Yes/No)**
4. If yes, provide:
   - Vulnerability type
   - Affected path IDs
   - Minimal fix suggestion
"""
        
        # 简化的上下文
        context_section = f"**Contract**: `{ctx['contract']}`\n**Function**: `{ctx['function_signature']}`\n**Risk Score**: {ctx['total_risk_score']:.1f}"
        
        # 路径摘要
        paths_summary = []
        for path in ctx['path_slices'][:3]:  # 只取前 3 条
            paths_summary.append(f"- Path `{path['slice_id']}`: {path['sink_type']} (score: {path['risk_score']:.1f})")
        paths_section = "## Paths\n" + "\n".join(paths_summary)
        
        question = "\n## Question\n**Is there an exploitable vulnerability? (Yes/No)**\n**If yes, explain the vulnerability type and provide a fix.**"
        
        full_prompt = f"{role}\n\n{instruction}\n\n{context_section}\n\n{paths_section}\n\n{question}"
        
        return VulnerabilityPrompt(
            contract=ctx['contract'],
            function=ctx['function_signature'],
            prompt=full_prompt,
            metadata={
                'total_risk_score': ctx['total_risk_score'],
                'num_slices': len(ctx['path_slices'])
            }
        )
    
    def save_prompts(self, prompts: List[VulnerabilityPrompt], output_dir: str):
        """保存提示到文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for prompt in prompts:
            # 文件名：contract_function.txt
            filename = f"{prompt.contract}_{prompt.function.replace('(', '_').replace(')', '').replace(',', '_')}.txt"
            filepath = output_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(prompt.prompt)
            
            logger.debug(f"[Prompt] 已保存: {filename}")
        
        # 同时保存元数据
        metadata_file = output_path / "_metadata.json"
        metadata = [
            {
                'contract': p.contract,
                'function': p.function,
                'filename': f"{p.contract}_{p.function.replace('(', '_').replace(')', '').replace(',', '_')}.txt",
                **p.metadata
            }
            for p in prompts
        ]
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[Prompt] 已保存 {len(prompts)} 个提示到: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prompt Constructor - 构造漏洞检测提示')
    parser.add_argument('distilled_file', help='蒸馏结果 JSON 文件')
    parser.add_argument('-o', '--output', help='输出目录',
                        default='cache/prompts')
    parser.add_argument('-s', '--source', help='原始 Solidity 源文件',
                        default=None)
    parser.add_argument('-l', '--language', choices=['zh', 'en'],
                        default='zh', help='提示语言')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # 构建提示
        builder = PromptBuilder(args.distilled_file, args.source)
        prompts = builder.build_all(language=args.language)
        
        # 保存提示
        builder.save_prompts(prompts, args.output)
        
        # 打印摘要
        print(f"\n✅ 提示构建完成!")
        print(f"   生成提示数量: {len(prompts)}")
        print(f"   输出目录: {args.output}")
        
        if prompts:
            top_prompt = max(prompts, key=lambda p: p.metadata['total_risk_score'])
            print(f"   最高风险函数: {top_prompt.function} (评分: {top_prompt.metadata['total_risk_score']:.1f})")
        
        return 0
    
    except Exception as e:
        logger.error(f"提示构建失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
