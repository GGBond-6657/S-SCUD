#!/usr/bin/env python3
"""
智能合约漏洞检测 - CKE/CKD 集成流程

使用方法：
    python scripts/vuln_detect_ckd.py <sol_file> [--model GPT_4_O_MINI]

示例：
    python scripts/vuln_detect_ckd.py test.sol
    python scripts/vuln_detect_ckd.py test.sol --top-k 3 --verbose
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path

# 添加项目根目录到路径
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# 导入项目模块
from camel.typing import ModelType
from camel.agents import ChatAgent

logger = logging.getLogger(__name__)


def run_cke(sol_file: str, output_dir: str, solc_version: str = None) -> str:
    """
    运行 Context Knowledge Extraction
    
    Returns:
        CKB JSONL 文件路径
    """
    from scripts.slither_ck_extractor import SlitherCKExtractor
    
    logger.info("[Pipeline] Step 1: Context Knowledge Extraction")
    
    ckb_file = Path(output_dir) / "ckb.jsonl"
    
    try:
        extractor = SlitherCKExtractor(sol_file, solc_version)
        ckb = extractor.extract()
        extractor.save_to_jsonl(ckb, str(ckb_file))
        
        logger.info(f"[Pipeline] ✅ CKE 完成: {ckb_file}")
        return str(ckb_file)
    
    except Exception as e:
        logger.error(f"[Pipeline] ❌ CKE 失败: {e}")
        raise


def run_ckd(ckb_file: str, output_dir: str, top_k: int = 5, threshold: float = 5.0) -> str:
    """
    运行 Context Knowledge Distillation
    
    Returns:
        蒸馏结果 JSON 文件路径
    """
    from scripts.slither_path_distill import PathDistiller
    
    logger.info("[Pipeline] Step 2: Context Knowledge Distillation")
    
    distilled_file = Path(output_dir) / "distilled.json"
    
    try:
        distiller = PathDistiller(ckb_file)
        contexts = distiller.distill(top_k=top_k, risk_threshold=threshold)
        distiller.save_to_json(contexts, str(distilled_file))
        
        logger.info(f"[Pipeline] ✅ CKD 完成: {distilled_file}")
        return str(distilled_file)
    
    except Exception as e:
        logger.error(f"[Pipeline] ❌ CKD 失败: {e}")
        raise


def build_prompts(distilled_file: str, output_dir: str, source_file: str = None, language: str = 'zh') -> list:
    """
    构建 LLM 提示
    
    Returns:
        提示列表
    """
    from scripts.prompt_builder import PromptBuilder
    
    logger.info("[Pipeline] Step 3: Prompt Construction")
    
    prompts_dir = Path(output_dir) / "prompts"
    
    try:
        builder = PromptBuilder(distilled_file, source_file)
        prompts = builder.build_all(language=language)
        builder.save_prompts(prompts, str(prompts_dir))
        
        logger.info(f"[Pipeline] ✅ Prompt 构建完成: {len(prompts)} 个")
        return prompts
    
    except Exception as e:
        logger.error(f"[Pipeline] ❌ Prompt 构建失败: {e}")
        raise


def llm_detect(prompts: list, model_type: ModelType, output_dir: str) -> dict:
    """
    使用 LLM 进行漏洞检测
    
    Returns:
        检测结果字典
    """
    logger.info("[Pipeline] Step 4: LLM Vulnerability Detection")
    
    results = {}
    
    for i, prompt_obj in enumerate(prompts, 1):
        logger.info(f"[Pipeline] 检测函数 {i}/{len(prompts)}: {prompt_obj.function}")
        
        try:
            # 创建 ChatAgent
            agent = ChatAgent(
                system_message="你是一位智能合约安全专家。",
                model_type=model_type,
            )
            
            # 发送提示
            response = agent.step(prompt_obj.prompt)
            
            # 提取回复
            answer = response.msg.content if hasattr(response.msg, 'content') else str(response.msg)
            
            results[prompt_obj.function] = {
                'contract': prompt_obj.contract,
                'function': prompt_obj.function,
                'risk_score': prompt_obj.metadata.get('total_risk_score', 0),
                'llm_response': answer,
                'status': 'success'
            }
            
            logger.info(f"[Pipeline]   ✅ 检测完成")
        
        except Exception as e:
            logger.error(f"[Pipeline]   ❌ 检测失败: {e}")
            results[prompt_obj.function] = {
                'contract': prompt_obj.contract,
                'function': prompt_obj.function,
                'status': 'failed',
                'error': str(e)
            }
    
    # 保存结果
    results_file = Path(output_dir) / "detection_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[Pipeline] ✅ 检测结果已保存: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='智能合约漏洞检测 - 基于 CKE/CKD 的路径切片分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/vuln_detect_ckd.py test.sol
  python scripts/vuln_detect_ckd.py test.sol --top-k 3 --model GPT_4
  python scripts/vuln_detect_ckd.py test.sol --no-llm  # 只提取路径，不调用 LLM
        """
    )
    
    parser.add_argument('sol_file', help='Solidity 源文件路径')
    parser.add_argument('--output', '-o', default='cache/ckd_analysis',
                        help='输出目录 (默认: cache/ckd_analysis)')
    parser.add_argument('--model', default='GPT_4_O_MINI',
                        choices=['GPT_3_5_TURBO', 'GPT_4', 'GPT_4_TURBO', 'GPT_4_O_MINI'],
                        help='LLM 模型选择')
    parser.add_argument('--top-k', type=int, default=5,
                        help='每个函数保留的最大路径数 (默认: 5)')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='风险阈值 (默认: 5.0)')
    parser.add_argument('--solc', default=None,
                        help='指定 solc 版本')
    parser.add_argument('--language', choices=['zh', 'en'], default='zh',
                        help='提示语言 (默认: zh)')
    parser.add_argument('--no-llm', action='store_true',
                        help='只提取和蒸馏，不调用 LLM')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 检查文件存在
    sol_file = Path(args.sol_file)
    if not sol_file.exists():
        logger.error(f"文件不存在: {args.sol_file}")
        return 1
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("智能合约漏洞检测 - CKE/CKD Pipeline")
    logger.info("="*60)
    logger.info(f"输入文件: {sol_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"模型: {args.model}")
    logger.info("="*60)
    
    try:
        # Step 1: CKE
        ckb_file = run_cke(str(sol_file), str(output_dir), args.solc)
        
        # Step 2: CKD
        distilled_file = run_ckd(ckb_file, str(output_dir), args.top_k, args.threshold)
        
        # Step 3: Build Prompts
        prompts = build_prompts(distilled_file, str(output_dir), str(sol_file), args.language)
        
        if not prompts:
            logger.warning("[Pipeline] ⚠️  未生成任何提示（可能没有高风险路径）")
            return 0
        
        # Step 4: LLM Detection
        if not args.no_llm:
            model_type = getattr(ModelType, args.model)
            results = llm_detect(prompts, model_type, str(output_dir))
            
            # 打印摘要
            print("\n" + "="*60)
            print("检测摘要")
            print("="*60)
            
            success_count = sum(1 for r in results.values() if r['status'] == 'success')
            print(f"成功检测: {success_count}/{len(results)}")
            
            # 检查是否有漏洞
            vuln_found = False
            for func, result in results.items():
                if result['status'] == 'success':
                    response = result['llm_response'].lower()
                    if '漏洞存在: 是' in response or 'vulnerability: yes' in response:
                        vuln_found = True
                        print(f"\n⚠️  发现漏洞: {func}")
                        print(f"   风险评分: {result['risk_score']:.1f}")
            
            if not vuln_found:
                print("\n✅ 未发现明显漏洞")
            
            print("="*60)
        else:
            logger.info("[Pipeline] 跳过 LLM 检测（--no-llm）")
        
        logger.info("[Pipeline] ✅ 全部完成！")
        return 0
    
    except Exception as e:
        logger.error(f"[Pipeline] ❌ 流程失败: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
