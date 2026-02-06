"""
将文本解析版 CKD 集成到现有 run_contract_intel 流程

使用方法：
在 chatdev/tools/contract_static.py 中调用 enhance_with_text_ckd()
"""

from pathlib import Path
from typing import Dict, List
import sys
import os

# 添加项目路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from scripts.text_based_ckd import (
    TextBasedCKExtractor,
    TextBasedPromptBuilder,
    PathContext
)


def enhance_with_text_ckd(
    source_code: str,
    slither_outputs: Dict[str, str],
    top_k: int = 5,
    risk_threshold: float = 5.0,
    language: str = 'zh'
) -> Dict[str, any]:
    """
    使用文本解析 CKD 增强现有的静态分析结果
    
    Args:
        source_code: Solidity 源代码
        slither_outputs: 现有的 Slither 输出
        top_k: 返回前 K 个高风险函数
        risk_threshold: 风险阈值
        language: Prompt 语言
    
    Returns:
        增强后的结果字典，包含：
        - ckd_contexts: 路径上下文列表
        - ckd_prompts: 生成的 Prompt 列表
        - ckd_summary: 摘要信息
    """
    try:
        # 1. 提取上下文
        extractor = TextBasedCKExtractor(source_code, slither_outputs)
        contexts = extractor.extract_contexts()
        
        # 2. 过滤高风险
        high_risk_contexts = [ctx for ctx in contexts if ctx.risk_score >= risk_threshold]
        high_risk_contexts = high_risk_contexts[:top_k]
        
        # 3. 生成 Prompts
        prompts = []
        if high_risk_contexts:
            builder = TextBasedPromptBuilder(source_code, high_risk_contexts)
            prompts = [builder.build_prompt(ctx, language=language) for ctx in high_risk_contexts]
        
        # 4. 生成摘要
        summary = {
            'total_functions_analyzed': len(contexts),
            'high_risk_functions': len(high_risk_contexts),
            'function_risk_scores': {
                ctx.function_name: ctx.risk_score
                for ctx in high_risk_contexts
            },
            'top_risk_function': high_risk_contexts[0].function_name if high_risk_contexts else None,
            'max_risk_score': high_risk_contexts[0].risk_score if high_risk_contexts else 0.0
        }
        
        return {
            'ckd_contexts': [
                {
                    'function_name': ctx.function_name,
                    'risk_score': ctx.risk_score,
                    'sensitive_patterns': [
                        {
                            'type': p.pattern_type,
                            'line': p.line_number,
                            'code': p.code_snippet
                        }
                        for p in ctx.sensitive_patterns
                    ],
                    'requires': ctx.requires,
                    'external_calls': ctx.external_calls
                }
                for ctx in high_risk_contexts
            ],
            'ckd_prompts': prompts,
            'ckd_summary': summary
        }
    
    except Exception as e:
        # 如果 CKD 失败，返回空结果（不影响原有流程）
        return {
            'ckd_contexts': [],
            'ckd_prompts': [],
            'ckd_summary': {
                'error': str(e),
                'total_functions_analyzed': 0,
                'high_risk_functions': 0
            }
        }


# 示例：如何集成到 contract_static.py
def example_integration():
    """
    在 chatdev/tools/contract_static.py 的 run_contract_intel 函数中添加：
    
    ```python
    from chatdev.tools.ckd_integration import enhance_with_text_ckd
    
    def run_contract_intel(...):
        # ... 现有代码 ...
        
        # 在返回 payload 之前，添加 CKD 分析
        ckd_results = enhance_with_text_ckd(
            source_code=formatted_source,
            slither_outputs={
                'slither_cfg': slither_enhanced_data.get('cfg', ''),
                'slither_function_summary': slither_enhanced_data.get('function_summary', ''),
                'slither_human_summary': slither_enhanced_data.get('human_summary', ''),
            },
            top_k=3,
            risk_threshold=5.0
        )
        
        # 将 CKD 结果添加到 payload
        payload.update({
            'ckd_contexts': ckd_results['ckd_contexts'],
            'ckd_prompts': ckd_results['ckd_prompts'],
            'ckd_summary': ckd_results['ckd_summary'],
        })
        
        return payload
    ```
    """
    pass


if __name__ == '__main__':
    # 测试集成
    test_code = '''
    pragma solidity ^0.8.0;
    contract Test {
        mapping(address => uint) public balances;
        
        function withdraw(uint amount) public {
            require(balances[msg.sender] >= amount);
            msg.sender.call{value: amount}("");
            balances[msg.sender] -= amount;
        }
    }
    '''
    
    result = enhance_with_text_ckd(test_code, {}, top_k=3, risk_threshold=5.0)
    
    print("CKD 集成测试结果：")
    print(f"  分析函数数: {result['ckd_summary']['total_functions_analyzed']}")
    print(f"  高风险函数数: {result['ckd_summary']['high_risk_functions']}")
    
    if result['ckd_contexts']:
        print(f"\n高风险函数：")
        for ctx in result['ckd_contexts']:
            print(f"  - {ctx['function_name']}: {ctx['risk_score']:.1f}")
    
    if result['ckd_prompts']:
        print(f"\n生成了 {len(result['ckd_prompts'])} 个 Prompt")
        print(f"第一个 Prompt 长度: {len(result['ckd_prompts'][0])} 字符")
