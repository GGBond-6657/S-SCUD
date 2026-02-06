#!/usr/bin/env python3
"""
混合漏洞检测流程 - 三阶段策略
阶段1: 静态规则筛选
阶段2: 风险画像评分  
阶段3: CKD + LLM精细分析
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from slither import Slither
from slither.core.declarations import Contract

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RiskProfile:
    """合约风险画像"""
    name: str
    contract: Contract  # 不会被序列化
    risk_score: float
    complexity_score: int
    sensitive_operations: List[str]
    unprotected_functions: List[str]
    vulnerability_indicators: List[str]
    
    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            'name': self.name,
            'risk_score': self.risk_score,
            'complexity_score': self.complexity_score,
            'sensitive_operations': self.sensitive_operations,
            'unprotected_functions': self.unprotected_functions,
            'vulnerability_indicators': self.vulnerability_indicators
        }


class HybridDetectionPipeline:
    """三阶段混合检测流程"""
    
    def __init__(self, sol_file: str, budget: str = 'medium', ablation_config: dict = None):
        """
        初始化检测流程
        
        Args:
            sol_file: Solidity文件路径
            budget: 预算级别 (low/medium/high)
            ablation_config: 消融实验配置（可选）
        """
        self.sol_file = Path(sol_file)
        self.budget = budget
        
        # 消融实验配置：优先使用参数，其次环境变量，最后默认值
        if ablation_config is None:
            # 尝试从环境变量读取
            import os
            config_path = os.environ.get('ABLATION_CONFIG_PATH')
            if config_path and Path(config_path).exists():
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    ablation_config = json.load(f)
                logger.info(f"从环境变量加载消融配置: {config_path}")
        
        self.config = ablation_config or {
            'static_filter': True,
            'complexity_scoring': True,
            'sensitive_ops_detection': True,
            'unprotected_check': True,
            'budget_strategy': True,
            'path_distillation': True,
            'whitelist_enabled': True,
            'internal_check_detection': True,
            'single_contract_protection': True,
        }
        
        # ===== 阈值参数配置 (可通过 ablation_config 传入) =====
        # 1. 复杂度阈值 - 判定合约是否"高复杂度"
        self.complexity_threshold = self.config.get('complexity_threshold', 50)
        
        # 2. 风险评分阈值 (目标选择) - 决定哪些合约进入阶段3深入分析
        self.target_risk_threshold = self.config.get('target_risk_threshold', 10.0)
        
        # 3. 风险评分阈值 (路径蒸馏) - 过滤低风险路径
        self.path_risk_threshold = self.config.get('path_risk_threshold', 3.0)
        
        logger.info(f"消融配置: {self.config}")
        logger.info(f"阈值参数: complexity={self.complexity_threshold}, "
                   f"target_risk={self.target_risk_threshold}, "
                   f"path_risk={self.path_risk_threshold}")
        self.slither = None

        
        logger.info(f"初始化检测流程: {sol_file}, 预算={budget}")
    
    def run(self) -> Dict:
        """运行完整检测流程"""
        import os
        
        # 保存当前工作目录
        original_cwd = os.getcwd()
        
        try:
            # 查找项目根目录（包含 package.json 或多个子目录的顶层目录）
            sol_dir = self.sol_file.parent
            project_root = self._find_project_root(sol_dir)
            
            if project_root and project_root.exists():
                # 切换到项目根目录（确保Slither能正确解析相对导入）
                os.chdir(project_root)
                logger.info(f"切换工作目录到项目根目录: {project_root}")
                # 使用相对于项目根的路径
                relative_path = self.sol_file.relative_to(project_root)
                
                # 在初始化Slither之前切换solc版本
                try:
                    from chatdev.tools.contract_static import (
                        _detect_solidity_version,
                        _get_available_solc_versions,
                        _select_best_solc_version,
                        _switch_solc_version
                    )
                    
                    full_ver, major_minor, prefix = _detect_solidity_version(self.sol_file)
                    available = _get_available_solc_versions()
                    
                    if available:
                        best_version = _select_best_solc_version(full_ver, major_minor, prefix, available)
                        if best_version:
                            _switch_solc_version(best_version)
                            logger.info(f"🔧 切换solc版本到: {best_version}")
                except Exception as ve:
                    logger.warning(f"版本切换失败: {ve}")
                
                self.slither = Slither(str(relative_path))
            else:
                # 如果找不到项目根，使用绝对路径（向后兼容）
                logger.warning(f"未找到项目根目录，使用绝对路径")
                self.slither = Slither(str(self.sol_file))
            
            # 阶段1: 静态规则筛选
            logger.info("\n" + "="*70)
            logger.info("阶段1: 静态规则筛选")
            logger.info("="*70)
            filtered_contracts = self._stage1_static_filter()
            
            # 阶段2: 风险画像评分
            logger.info("\n" + "="*70)
            logger.info("阶段2: 风险画像评分")
            logger.info("="*70)
            risk_profiles = self._stage2_risk_scoring(filtered_contracts)
            
            # 阶段3: 选择目标并分析
            logger.info("\n" + "="*70)
            logger.info("阶段3: CKD + LLM精细分析")
            logger.info("="*70)
            
            # 消融点：检查是否跳过阶段3（C_no_ckd_cke实验）
            if self.config.get('skip_stage3', False):
                logger.warning("⚠️  阶段3已跳过（消融实验：C_no_ckd_cke）")
                logger.info("仅使用阶段1+2的静态分析和风险评分结果")
                target_profiles = []  # 空列表，表示未选择目标
                results = []
            else:
                target_profiles = self._select_targets(risk_profiles)
                results = self._stage3_detailed_analysis(target_profiles)
            
            return {
                'file': str(self.sol_file),
                'total_contracts': len(self.slither.contracts),
                'filtered_contracts': len(filtered_contracts),
                'analyzed_contracts': len(target_profiles),
                'profiles': [p.to_dict() for p in risk_profiles],
                'results': results
            }
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(original_cwd)
            logger.info(f"恢复工作目录到: {original_cwd}")
    
    def _find_project_root(self, start_dir: Path) -> Optional[Path]:
        """
        查找 Solidity 项目的根目录
        策略：向上查找包含 package.json 或具有典型项目结构的目录
        """
        current = start_dir
        candidates = []
        
        # 向上查找最多5层
        for level in range(5):
            # 策略1: 检查是否有 package.json（最可靠）
            if (current / 'package.json').exists():
                logger.info(f"找到 package.json，项目根目录: {current}")
                return current
            
            # 策略2: 检查是否有多个顶层模块目录（如 access/, token/, utils/）
            # 必须有至少3个子目录，且每个都包含 .sol 文件
            sol_subdirs = [d for d in current.iterdir() 
                          if d.is_dir() and list(d.glob('*.sol'))]
            
            if len(sol_subdirs) >= 3:
                # 进一步验证：这些子目录的名称应该是常见的模块名
                common_modules = {'access', 'token', 'utils', 'governance', 
                                 'finance', 'proxy', 'interfaces', 'security',
                                 'metatx', 'crosschain', 'account', 'mocks'}
                dir_names = {d.name.lower() for d in sol_subdirs}
                matches = len(dir_names & common_modules)
                
                if matches >= 2:  # 至少匹配2个常见模块名
                    candidates.append((current, matches, len(sol_subdirs)))
            
            # 向上一层
            parent = current.parent
            if parent == current:  # 已到达文件系统根目录
                break
            current = parent
        
        # 选择最佳候选（匹配度最高的）
        if candidates:
            best = max(candidates, key=lambda x: (x[1], x[2]))  # 先按匹配度，再按子目录数量
            logger.info(f"找到项目根目录（{best[1]}个模块匹配，{best[2]}个子目录）: {best[0]}")
            return best[0]
        
        # 未找到，返回文件所在目录的父目录作为备选
        logger.warning(f"未找到明确的项目根目录，使用父目录: {start_dir.parent}")
        return start_dir.parent
    
    def _stage1_static_filter(self) -> List[Contract]:
        """阶段1: 静态规则筛选"""
        contracts = self.slither.contracts_derived or self.slither.contracts
        
        logger.info(f"总合约数: {len(contracts)}")
        
        # 消融点: 如果禁用静态筛选，直接返回全部合约
        if not self.config['static_filter']:
            logger.info(f"  ⚠️  静态筛选已禁用，保留全部 {len(contracts)} 个合约")
            return contracts
        
        filtered = []
        
        # 特殊规则: 如果文件只有一个合约，直接保留，不进行过滤
        if len(contracts) == 1 and self.config['single_contract_protection']:
            logger.info(f"  ℹ️  文件只有一个合约: {contracts[0].name}，跳过过滤直接保留")
            return contracts
        
        for contract in contracts:
            # 规则1: 排除接口和库
            if contract.is_interface or contract.is_library:
                logger.info(f"  ❌ 排除: {contract.name} (接口/库)")
                continue
            
            # 规则2: 排除明显的工具合约
            tool_names = ['SafeMath', 'Math', 'Address', 'Strings', 'Context', 
                         'Bytes', 'Arrays', 'Counters', 'EnumerableSet', 'EnumerableMap']
            if contract.name in tool_names:
                logger.info(f"  ❌ 排除: {contract.name} (工具合约)")
                continue
            
            # 规则3: 排除纯只读合约（没有状态变量且没有状态修改）
            has_state = len(contract.state_variables) > 0
            has_state_modification = any(
                f.is_implemented and (f.state_variables_written or not f.view and not f.pure)
                for f in contract.functions
            )
            
            if not has_state and not has_state_modification:
                logger.info(f"  ❌ 排除: {contract.name} (纯只读合约)")
                continue
            
            # 规则4: 排除抽象合约（无任何实现函数）
            if contract.functions and not any(f.is_implemented for f in contract.functions):
                logger.info(f"  ❌ 排除: {contract.name} (抽象合约)")
                continue
            
            logger.info(f"  ✅ 保留: {contract.name}")
            filtered.append(contract)
        
        # 保底规则: 如果所有合约都被过滤了，保留原始列表中的第一个非抽象合约
        if not filtered and contracts:
            for contract in contracts:
                # 找第一个有实现的合约
                if contract.functions and any(f.is_implemented for f in contract.functions):
                    logger.info(f"  ⚠️  所有合约都被过滤，保底保留: {contract.name}")
                    filtered.append(contract)
                    break
            # 如果还是没有，就保留第一个
            if not filtered:
                logger.info(f"  ⚠️  所有合约都被过滤，保底保留第一个: {contracts[0].name}")
                filtered.append(contracts[0])
        
        logger.info(f"\n筛选后剩余: {len(filtered)} 个合约")
        return filtered
    
    def _stage2_risk_scoring(self, contracts: List[Contract]) -> List[RiskProfile]:
        """阶段2: 风险画像评分"""
        profiles = []
        
        for contract in contracts:
            profile = self._calculate_risk_profile(contract)
            profiles.append(profile)
            
            logger.info(f"\n合约: {profile.name}")
            logger.info(f"  风险评分: {profile.risk_score:.1f}")
            logger.info(f"  复杂度: {profile.complexity_score}")
            logger.info(f"  敏感操作: {len(profile.sensitive_operations)}")
            logger.info(f"  无保护函数: {len(profile.unprotected_functions)}")
            if profile.vulnerability_indicators:
                logger.info(f"  漏洞指标:")
                for indicator in profile.vulnerability_indicators[:3]:
                    logger.info(f"    - {indicator}")
        
        # 按风险评分排序
        profiles.sort(key=lambda x: x.risk_score, reverse=True)
        
        logger.info(f"\n风险排序（Top-5）:")
        for i, p in enumerate(profiles[:5], 1):
            logger.info(f"  {i}. {p.name}: {p.risk_score:.1f}")
        
        return profiles
    
    def _calculate_risk_profile(self, contract: Contract) -> RiskProfile:
        """计算单个合约的风险画像"""
        risk_score = 0.0
        complexity_score = 0
        sensitive_ops = []
        unprotected_funcs = []
        indicators = []
        
        # 消融点: 复杂度评分
        if self.config['complexity_scoring']:
            complexity_score += len(contract.state_variables) * 3
            complexity_score += len([f for f in contract.functions if f.is_implemented]) * 2
            complexity_score += len(contract.modifiers) * 2
        
        # 检查敏感操作
        for func in contract.functions:
            if not func.is_implemented:
                continue
            
            # 检查节点中的敏感操作
            for node in func.nodes:
                expr = str(node.expression) if node.expression else ""
                
                # 消融点: 敏感操作检测
                if self.config['sensitive_ops_detection']:
                    if 'delegatecall' in expr:
                        sensitive_ops.append(f"{func.name}: delegatecall")
                        risk_score += 8.0
                        indicators.append(f"⚠️ {func.name}: 使用delegatecall")
                    
                    if any(kw in expr for kw in ['call.value', 'call{value:']):
                        sensitive_ops.append(f"{func.name}: call.value")
                        risk_score += 10.0
                        indicators.append(f"⚠️ {func.name}: 以太转账")
                    
                    if 'selfdestruct' in expr:
                        sensitive_ops.append(f"{func.name}: selfdestruct")
                        risk_score += 15.0
                        indicators.append(f"⚠️ {func.name}: 合约自毁")
            
            # 消融点: 无保护函数检测
            if self.config['unprotected_check']:
                # 检查访问控制
                if func.visibility in ['public', 'external']:
                    if not func.modifiers:
                        # 检查是否修改状态
                        if func.state_variables_written:
                            # 消融点: 白名单机制
                            if self.config['whitelist_enabled']:
                                # 扩大白名单：包括 ERC20/ERC721/ERC1155 标准函数和常见安全函数
                                safe_names = [
                                    # ERC20 标准
                                    'transfer', 'transferFrom', 'approve', 'increaseAllowance', 'decreaseAllowance',
                                    'mint', 'burn', 'burnFrom',
                                    # ERC721 标准
                                    'safeTransferFrom', 'setApprovalForAll', 'approve',
                                    # ERC1155 标准  
                                    'safeBatchTransferFrom', 'setApprovalForAll',
                                    # 常见构造函数
                                    'constructor', 'initialize', 'init',
                                    # Fallback/Receive
                                    'fallback', 'receive',
                                    # 访问控制相关（内部有检查）
                                    'grantRole', 'revokeRole', 'renounceRole',
                                    # Pausable
                                    'pause', 'unpause',
                                    # 常见公开函数
                                    'deposit', 'withdraw', 'stake', 'unstake',
                                    'vote', 'delegate', 'execute', 'propose'
                                ]
                                # 检查函数名是否在白名单中（不区分大小写）
                                func_name_lower = func.name.lower()
                                if func_name_lower in [s.lower() for s in safe_names]:
                                    continue  # 白名单函数，跳过检查
                            
                            # 消融点: 内部检查保护检测
                            if self.config['internal_check_detection']:
                                # 额外检查：函数内部是否有 require/assert/revert 保护
                                has_internal_check = any(
                                    'require' in str(node.expression) or 
                                    'assert' in str(node.expression) or
                                    'revert' in str(node.expression) or
                                    'msg.sender' in str(node.expression)  # 常见的访问控制模式
                                    for node in func.nodes if node.expression
                                )
                                
                                if has_internal_check:
                                    continue  # 有内部检查，跳过
                            
                            unprotected_funcs.append(func.name)
                            risk_score += 1.5  # 进一步降低评分：从 2.0 改为 1.5
                            indicators.append(f"⚠️ {func.name}: 状态修改缺少明显保护")
        
        # 去重
        sensitive_ops = list(set(sensitive_ops))
        unprotected_funcs = list(set(unprotected_funcs))
        
        return RiskProfile(
            name=contract.name,
            contract=contract,
            risk_score=risk_score,
            complexity_score=complexity_score,
            sensitive_operations=sensitive_ops,
            unprotected_functions=unprotected_funcs,
            vulnerability_indicators=indicators
        )
    
    def _select_targets(self, profiles: List[RiskProfile]) -> List[RiskProfile]:
        """
        根据预算选择分析目标
        改进：即使低风险，也保留部分合约用于分析
        """
        if not profiles:
            return []
        
        # 消融点: 预算策略
        if not self.config['budget_strategy']:
            logger.info("⚠️  预算策略已禁用，全量分析所有合约")
            return profiles
        
        if self.budget == 'low':
            # 只分析Top-1，但至少保留1个
            targets = [profiles[0]]
        elif self.budget == 'medium':
            # 分析Top-3或风险>=target_risk_threshold的合约
            # 改进：如果所有合约风险都很低，也保留至少1-2个
            high_risk = [p for p in profiles if p.risk_score >= self.target_risk_threshold]
            if high_risk:
                targets = high_risk[:3] if len(high_risk) >= 3 else high_risk
            else:
                # 如果没有高风险合约，选择风险最高的1-2个
                logger.info("⚠️ 未发现高风险合约，选择风险最高的1-2个进行分析")
                targets = profiles[:min(2, len(profiles))]
        else:  # high
            # 全量分析
            targets = profiles
        
        logger.info(f"\n根据预算'{self.budget}'选择 {len(targets)} 个合约进行深入分析:")
        for p in targets:
            logger.info(f"  - {p.name} (风险={p.risk_score:.1f})")
        
        return targets
    
    def _stage3_detailed_analysis(self, profiles: List[RiskProfile]) -> List[Dict]:
        """阶段3: 对选定合约执行CKE+CKD分析"""
        from scripts.slither_ck_extractor import SlitherCKExtractor
        from scripts.slither_path_distill import PathDistiller
        
        results = []
        
        # 创建临时目录存储CKE结果
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix='ckd_'))
        
        try:
            for profile in profiles:
                logger.info(f"\n分析合约: {profile.name}")
                
                # Step 1: CKE - 提取合约知识
                logger.info(f"  [CKE] 提取合约知识...")
                try:
                    extractor = SlitherCKExtractor(str(self.sol_file))
                    # 传递profile.contract作为目标合约
                    ckb = extractor.extract(target_contract=profile.contract)
                    
                    ckb_file = temp_dir / f"{profile.name}_ckb.jsonl"
                    extractor.save_to_jsonl(ckb, str(ckb_file))
                    logger.info(f"  [CKE] ✅ 知识库生成完成")
                    
                    # Step 2: CKD - 路径蒸馏
                    logger.info(f"  [CKD] 路径蒸馏...")
                    
                    # 消融点: 路径蒸馏
                    if self.config['path_distillation']:
                        distiller = PathDistiller(str(ckb_file))
                        # 使用可配置的 path_risk_threshold: 过滤低风险路径
                        distilled_contexts = distiller.distill(top_k=5, risk_threshold=self.path_risk_threshold)
                    else:
                        # 禁用路径蒸馏：使用完整CKE结果，不进行路径筛选
                        logger.info(f"  [CKD] ⚠️  路径蒸馏已禁用，使用完整CKE数据（无筛选）")
                        distiller = PathDistiller(str(ckb_file))
                        # 不设置risk_threshold和top_k限制，返回所有路径
                        distilled_contexts = distiller.distill(top_k=999, risk_threshold=0.0)
                    
                    # 汇总蒸馏结果
                    ckd_results = {
                        'total_paths': sum(len(ctx.path_slices) for ctx in distilled_contexts),
                        'high_risk_functions': len(distilled_contexts),
                        'path_details': []
                    }
                    
                    # 消融点：根据path_distillation决定函数数量限制
                    max_functions = 3 if self.config['path_distillation'] else len(distilled_contexts)
                    
                    for ctx in distilled_contexts[:max_functions]:
                        # 提取函数级元数据（所有path共享）
                        # 添加 None 检查，避免 'NoneType' object has no attribute 'name' 错误
                        if ctx.path_slices and ctx.path_slices[0] and hasattr(ctx.path_slices[0], 'function'):
                            func_name = ctx.path_slices[0].function if ctx.path_slices[0].function else 'unknown'
                        else:
                            func_name = 'unknown'
                        
                        function_metadata = {
                            'function': func_name,
                            'function_signature': ctx.function_signature if hasattr(ctx, 'function_signature') else '',
                            'visibility': ctx.visibility if hasattr(ctx, 'visibility') else '',
                            'modifiers': ctx.modifiers if hasattr(ctx, 'modifiers') else [],
                            'state_var_definitions': ctx.state_var_definitions if hasattr(ctx, 'state_var_definitions') else {},
                            'dependent_functions': ctx.dependent_function_code if hasattr(ctx, 'dependent_function_code') else {},
                        }
                        
                        # 消融点：根据path_distillation决定路径数量限制
                        max_paths = 2 if self.config['path_distillation'] else len(ctx.path_slices)
                        
                        for path in ctx.path_slices[:max_paths]:
                            # 添加 None 检查，避免访问 None 对象的属性
                            if path:
                                ckd_results['path_details'].append({
                                    **function_metadata,  # 展开函数级元数据
                                    'sink_type': getattr(path, 'sink_type', 'unknown'),
                                    'risk_score': getattr(path, 'risk_score', 0),
                                    'risk_factors': getattr(path, 'risk_factors', []),
                                    'guards': getattr(path, 'guards', []),
                                    'state_writes': getattr(path, 'state_vars_written', []),
                                    'state_reads': getattr(path, 'state_vars_read', []),
                                    'dependent_function_list': getattr(path, 'dependent_functions', [])
                                })
                    
                    logger.info(f"  [CKD] ✅ 发现 {ckd_results['total_paths']} 条可疑路径")
                    
                except Exception as e:
                    logger.warning(f"  [CKE/CKD] ⚠️  分析失败: {e}")
                    ckd_results = {'error': str(e)}
                
                # 整合结果
                result = {
                    'contract': getattr(profile, 'name', 'unknown'),
                    'risk_score': getattr(profile, 'risk_score', 0),
                    'sensitive_operations': getattr(profile, 'sensitive_operations', []),
                    'unprotected_functions': getattr(profile, 'unprotected_functions', []),
                    'vulnerability_indicators': getattr(profile, 'vulnerability_indicators', []),
                    'recommendation': self._generate_recommendation(profile) if profile else '',
                    'ckd_analysis': ckd_results  # 添加CKD分析结果
                }
                
                results.append(result)
                logger.info(f"  ✅ 分析完成")
        
        finally:
            # 清理临时文件
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return results
    
    def _generate_recommendation(self, profile: RiskProfile) -> str:
        """生成分析建议"""
        # 提高阈值降低误报：20→30, 10→15
        if profile.risk_score >= 30:
            return "🔴 高风险：强烈建议人工审计"
        elif profile.risk_score >= 15:
            return "🟡 中风险：建议重点关注敏感操作"
        else:
            return "🟢 低风险：可进行常规检查"
    
    def _detect_solc_version(self) -> Optional[str]:
        """自动检测Solidity版本"""
        import re
        
        try:
            content = self.sol_file.read_text(encoding='utf-8')
            pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', content)
            
            if pragma_match:
                version_spec = pragma_match.group(1).strip()
                logger.info(f"检测到版本声明: {version_spec}")
                
                # 提取版本号
                if '^' in version_spec:
                    version = version_spec.replace('^', '').strip()
                elif '>=' in version_spec:
                    version = version_spec.replace('>=', '').strip().split()[0]
                else:
                    version = version_spec
                
                return version
        except Exception as e:
            logger.warning(f"版本检测失败: {e}")
        
        return None


def main():
    """测试入口"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python hybrid_detection_pipeline.py <sol_file> [budget]")
        print("  budget: low | medium | high (默认: medium)")
        sys.exit(1)
    
    sol_file = sys.argv[1]
    budget = sys.argv[2] if len(sys.argv) > 2 else 'medium'
    
    pipeline = HybridDetectionPipeline(sol_file, budget)
    results = pipeline.run()
    
    # 保存结果
    output_file = Path('cache/hybrid_detection_result.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n结果已保存到: {output_file}")
    
    # 打印摘要
    print("\n" + "="*70)
    print("检测摘要")
    print("="*70)
    print(f"总合约数: {results['total_contracts']}")
    print(f"筛选后: {results['filtered_contracts']}")
    print(f"深入分析: {results['analyzed_contracts']}")
    print(f"\n发现的问题:")
    for result in results['results']:
        print(f"\n合约: {result['contract']}")
        print(f"  风险评分: {result['risk_score']:.1f}")
        print(f"  建议: {result['recommendation']}")
        if result['vulnerability_indicators']:
            print(f"  漏洞指标:")
            for indicator in result['vulnerability_indicators'][:3]:
                print(f"    {indicator}")


if __name__ == "__main__":
    main()
