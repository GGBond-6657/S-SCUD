#!/usr/bin/env python3
"""
Context Knowledge Extraction (CKE) Module
基于 Slither 提取智能合约的结构化知识库（Type, CFG, DFG）

功能：
1. Type Information: 继承关系、函数可见性、修饰符
2. Control-Flow: CFG 构建，识别敏感 Sink 节点
3. Data-Flow: SSA 分析、状态变量读写依赖
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# 尝试导入 Slither API，如果未安装则提供友好提示
try:
    from slither import Slither
    from slither.core.declarations import Contract, Function
    from slither.core.cfg.node import Node, NodeType
    from slither.core.variables.state_variable import StateVariable
    from slither.slithir.operations import (
        HighLevelCall, LowLevelCall, InternalCall, LibraryCall,
        SolidityCall, Assignment, Binary, Index
    )
    SLITHER_AVAILABLE = True
except ImportError:
    SLITHER_AVAILABLE = False
    print("⚠️  Slither Python API 未安装，请运行: pip install slither-analyzer")


logger = logging.getLogger(__name__)


class SinkType(Enum):
    """敏感操作类型（Sink 节点）"""
    CALL_VALUE = "call.value"           # 以太转账
    SELFDESTRUCT = "selfdestruct"       # 自毁合约
    DELEGATECALL = "delegatecall"       # 代理调用
    STATE_WRITE = "state_write"         # 状态变量写入
    EXTERNAL_CALL = "external_call"     # 外部调用
    UNCHECKED_CALL = "unchecked_call"   # 未检查返回值的调用


@dataclass
class TypeInfo:
    """类型信息"""
    contract_name: str
    inheritance: List[str]
    functions: Dict[str, Dict]  # {func_name: {visibility, modifiers, state_mutability}}
    state_vars: Dict[str, Dict]  # {var_name: {type, visibility}}


@dataclass
class CFGNode:
    """CFG 节点"""
    node_id: int
    node_type: str
    expression: str
    source_code: str
    is_sink: bool
    sink_type: Optional[str]
    successors: List[int]
    dominators: List[int]  # 支配节点（用于后向切片）


@dataclass
class DataFlowInfo:
    """数据流信息"""
    variable: str
    definition_nodes: List[int]  # 定义该变量的节点
    use_nodes: List[int]         # 使用该变量的节点
    is_state_var: bool
    depends_on: List[str]        # 依赖的其他变量


@dataclass
class FunctionKnowledge:
    """函数级知识"""
    contract: str
    function: str
    signature: str
    visibility: str
    modifiers: List[str]
    state_mutability: str
    cfg_nodes: List[CFGNode]
    sink_nodes: List[int]  # Sink 节点 ID 列表
    data_flow: List[DataFlowInfo]
    state_reads: List[str]
    state_writes: List[str]
    internal_calls: List[str]  # 调用的内部函数


@dataclass
class ContractKnowledgeBase:
    """合约知识库（CKB）"""
    source_file: str
    type_info: TypeInfo
    functions: List[FunctionKnowledge]


class SlitherCKExtractor:
    """Slither 上下文知识提取器"""
    
    def __init__(self, sol_file: str, solc_version: Optional[str] = None):
        """
        初始化提取器
        
        Args:
            sol_file: Solidity 源文件路径
            solc_version: 指定 solc 版本（可选，将自动切换）
        """
        if not SLITHER_AVAILABLE:
            raise RuntimeError("Slither 未安装，请运行: pip install slither-analyzer")
        
        self.sol_file = Path(sol_file)
        if not self.sol_file.exists():
            raise FileNotFoundError(f"文件不存在: {sol_file}")
        
        logger.info(f"[CKE] 开始分析: {self.sol_file.name}")
        
        # 初始化 Slither（切换到项目根目录以正确解析相对导入）
        import os
        original_cwd = os.getcwd()
        
        try:
            sol_dir = self.sol_file.parent
            project_root = self._find_project_root(sol_dir)
            
            if project_root and project_root.exists():
                os.chdir(project_root)
                logger.info(f"[CKE] 切换工作目录到项目根目录: {project_root}")
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
                            logger.info(f"[CKE] 🔧 切换solc版本到: {best_version}")
                except Exception as ve:
                    logger.warning(f"[CKE] 版本切换失败: {ve}")
                
                self.slither = Slither(str(relative_path))
            else:
                logger.warning(f"[CKE] 未找到项目根目录，使用绝对路径")
                self.slither = Slither(str(self.sol_file))
        except Exception as e:
            raise RuntimeError(f"Slither 初始化失败: {e}")
        finally:
            os.chdir(original_cwd)
            logger.info(f"[CKE] 恢复工作目录到: {original_cwd}")
    
    def _find_project_root(self, start_dir: Path) -> Path:
        """查找 Solidity 项目根目录"""
        current = start_dir
        candidates = []
        
        for _ in range(5):
            # 策略1: package.json
            if (current / 'package.json').exists():
                return current
            
            # 策略2: 至少3个包含 .sol 的子目录，且匹配常见模块名
            sol_subdirs = [d for d in current.iterdir() 
                          if d.is_dir() and list(d.glob('*.sol'))]
            
            if len(sol_subdirs) >= 3:
                common_modules = {'access', 'token', 'utils', 'governance', 
                                 'finance', 'proxy', 'interfaces', 'security',
                                 'metatx', 'crosschain', 'account', 'mocks'}
                dir_names = {d.name.lower() for d in sol_subdirs}
                matches = len(dir_names & common_modules)
                
                if matches >= 2:
                    candidates.append((current, matches, len(sol_subdirs)))
            
            parent = current.parent
            if parent == current:
                break
            current = parent
        
        if candidates:
            best = max(candidates, key=lambda x: (x[1], x[2]))
            return best[0]
        
        return start_dir.parent
    
    def extract(self, target_contract=None) -> ContractKnowledgeBase:
        """
        提取完整的合约知识库
        
        Args:
            target_contract: 指定要分析的合约对象（可选）
                           如果为None，则自动选择主合约
        
        Returns:
            ContractKnowledgeBase 对象
        """
        # 如果指定了目标合约，直接使用
        if target_contract is not None:
            main_contract = target_contract
            logger.info(f"[CKE] 使用指定合约: {main_contract.name}")
        else:
            # 提取所有合约（通常取主合约）
            contracts = self.slither.contracts_derived
            if not contracts:
                contracts = self.slither.contracts
            
            if not contracts:
                raise ValueError("未找到合约定义")
            
            # 选择主合约：优先选择非接口、非库的实际合约
            # 使用复杂度评分选择最有可能是主合约的
            candidates = []
            for contract in contracts:
                # 跳过接口和库
                if contract.is_interface or contract.is_library:
                    continue
                # 跳过没有实现的合约
                if not contract.functions or not any(f.is_implemented for f in contract.functions):
                    continue
                
                # 计算复杂度评分
                score = 0
                score += len(contract.state_variables) * 3  # 状态变量权重高
                score += len([f for f in contract.functions if f.is_implemented]) * 2  # 实现的函数
                score += len(contract.modifiers) * 2  # 修饰符
                
                # 检查是否有敏感操作（call, delegatecall, selfdestruct等）
                has_sensitive = False
                for func in contract.functions:
                    if not func.is_implemented:
                        continue
                    for node in func.nodes:
                        expr = str(node.expression) if node.expression else ""
                        if any(keyword in expr for keyword in ['call', 'delegatecall', 'selfdestruct', 'transfer', 'send']):
                            has_sensitive = True
                            break
                    if has_sensitive:
                        break
                
                if has_sensitive:
                    score += 10  # 有敏感操作的合约优先级更高
                
                candidates.append((contract, score))
            
            # 按评分排序，选择最高分的
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                main_contract = candidates[0][0]
                logger.info(f"[CKE] 候选合约: {[(c.name, s) for c, s in candidates[:3]]}")
            else:
                # 如果没找到实现合约，回退到第一个合约
                main_contract = contracts[0]
                logger.warning(f"[CKE] 未找到实现合约，使用: {main_contract.name}")
            
            logger.info(f"[CKE] 自动选择合约: {main_contract.name} (函数数={len(main_contract.functions)}, 状态变量={len(main_contract.state_variables)})")
        
        # 统一输出分析目标
        logger.info(f"[CKE] 开始提取知识: {main_contract.name}")        # 1. 提取类型信息
        type_info = self._extract_type_info(main_contract)
        
        # 2. 提取所有函数的知识
        functions_knowledge = []
        for func in main_contract.functions:
            # 跳过构造函数、回退函数等特殊函数（可根据需要调整）
            if func.is_constructor or func.is_fallback or func.is_receive:
                continue
            
            func_knowledge = self._extract_function_knowledge(main_contract, func)
            functions_knowledge.append(func_knowledge)
        
        logger.info(f"[CKE] 提取了 {len(functions_knowledge)} 个函数的知识")
        
        return ContractKnowledgeBase(
            source_file=str(self.sol_file),
            type_info=type_info,
            functions=functions_knowledge
        )
    
    def _extract_type_info(self, contract: Contract) -> TypeInfo:
        """提取类型信息"""
        # 继承关系
        inheritance = [base.name for base in contract.inheritance]
        
        # 函数信息
        functions = {}
        for func in contract.functions:
            # 获取状态可变性（兼容不同 Slither 版本）
            state_mutability = self._get_state_mutability(func)
            
            functions[func.name] = {
                "visibility": func.visibility,
                "modifiers": [mod.name for mod in func.modifiers],
                "state_mutability": state_mutability,
                "is_implemented": func.is_implemented,
            }
        
        # 状态变量
        state_vars = {}
        for var in contract.state_variables:
            state_vars[var.name] = {
                "type": str(var.type),
                "visibility": var.visibility,
            }
        
        return TypeInfo(
            contract_name=contract.name,
            inheritance=inheritance,
            functions=functions,
            state_vars=state_vars
        )
    
    def _get_state_mutability(self, func: Function) -> str:
        """获取函数的状态可变性（兼容不同 Slither 版本）"""
        # 尝试直接获取 state_mutability 属性
        if hasattr(func, 'state_mutability'):
            return func.state_mutability
        
        # 否则根据其他属性推断
        if getattr(func, 'pure', False):
            return 'pure'
        elif getattr(func, 'view', False):
            return 'view'
        elif getattr(func, 'payable', False):
            return 'payable'
        else:
            return 'nonpayable'
    
    def _extract_function_knowledge(self, contract: Contract, func: Function) -> FunctionKnowledge:
        """提取单个函数的完整知识"""
        logger.debug(f"[CKE] 提取函数: {func.name}")
        
        # 构建 CFG 节点列表
        cfg_nodes = []
        sink_nodes = []
        
        for node in func.nodes:
            cfg_node, is_sink, sink_type = self._build_cfg_node(node)
            cfg_nodes.append(cfg_node)
            
            if is_sink:
                sink_nodes.append(cfg_node.node_id)
                logger.info(f"  🎯 发现 Sink: {sink_type} at node {cfg_node.node_id}, 表达式: {cfg_node.expression}")
        
        # 提取数据流信息
        data_flow = self._extract_data_flow(func)
        
        # 识别状态变量读写
        state_reads, state_writes = self._identify_state_access(func)
        
        # 识别内部函数调用
        internal_calls = self._extract_internal_calls(func)
        
        return FunctionKnowledge(
            contract=contract.name,
            function=func.name,
            signature=func.full_name,
            visibility=func.visibility,
            modifiers=[mod.name for mod in func.modifiers],
            state_mutability=self._get_state_mutability(func),
            cfg_nodes=cfg_nodes,
            sink_nodes=sink_nodes,
            data_flow=data_flow,
            state_reads=list(state_reads),
            state_writes=list(state_writes),
            internal_calls=internal_calls
        )
    
    def _build_cfg_node(self, node: Node) -> Tuple[CFGNode, bool, Optional[str]]:
        """
        构建 CFG 节点，并识别是否为 Sink
        
        Returns:
            (CFGNode, is_sink, sink_type)
        """
        # 判断是否为 Sink 节点
        is_sink = False
        sink_type = None
        
        # 优先检查高危操作（IR层面更准确）
        for ir in node.irs:
            if isinstance(ir, LowLevelCall):
                # 检查是否是delegatecall
                if hasattr(ir, 'function_name'):
                    func_name = str(ir.function_name).lower()
                    if 'delegatecall' in func_name:
                        is_sink = True
                        sink_type = SinkType.DELEGATECALL.value
                        logger.debug(f"    通过IR识别delegatecall: {ir}")
                        break
                    elif 'call' in func_name:
                        is_sink = True
                        sink_type = SinkType.UNCHECKED_CALL.value
                        break
            elif isinstance(ir, HighLevelCall):
                if not is_sink:
                    is_sink = True
                    sink_type = SinkType.EXTERNAL_CALL.value
        
        # 如果IR没识别出来，再检查表达式字符串
        if not is_sink and node.type == NodeType.EXPRESSION:
            expr = str(node.expression) if node.expression else ""
            
            if "delegatecall" in expr:
                is_sink = True
                sink_type = SinkType.DELEGATECALL.value
                logger.debug(f"    通过表达式识别delegatecall: {expr[:60]}")
            elif "call.value" in expr or "call{value:" in expr:
                is_sink = True
                sink_type = SinkType.CALL_VALUE.value
            elif "selfdestruct" in expr:
                is_sink = True
                sink_type = SinkType.SELFDESTRUCT.value
        
        # 状态变量写入作为最低优先级（不覆盖前面的高危操作）
        if not is_sink and node.state_variables_written:
            is_sink = True
            sink_type = SinkType.STATE_WRITE.value
        
        # 计算支配节点（简化版，使用前驱节点代替）
        # 注意：Slither 的 Node 对象可能没有 dominators 属性
        dominators = []
        if hasattr(node, 'dominators'):
            dominators = [pred.node_id for pred in node.dominators]
        elif hasattr(node, 'fathers'):
            # 使用前驱节点（fathers）作为近似
            dominators = [pred.node_id for pred in node.fathers]
        
        cfg_node = CFGNode(
            node_id=node.node_id,
            node_type=str(node.type),
            expression=str(node.expression) if node.expression else "",
            source_code=str(node),
            is_sink=is_sink,
            sink_type=sink_type,
            successors=[son.node_id for son in node.sons],
            dominators=dominators
        )
        
        return cfg_node, is_sink, sink_type
    
    def _extract_data_flow(self, func: Function) -> List[DataFlowInfo]:
        """提取数据流信息（基于 SSA）"""
        data_flow = []
        
        # 收集所有读写的变量
        var_info = {}
        
        for node in func.nodes:
            # 读取的变量
            for var in node.variables_read:
                if var is None or not hasattr(var, 'name'):
                    continue
                var_name = var.name
                if var_name not in var_info:
                    var_info[var_name] = {
                        'defs': [],
                        'uses': [],
                        'is_state': isinstance(var, StateVariable)
                    }
                var_info[var_name]['uses'].append(node.node_id)
            
            # 写入的变量
            for var in node.variables_written:
                if var is None or not hasattr(var, 'name'):
                    continue
                var_name = var.name
                if var_name not in var_info:
                    var_info[var_name] = {
                        'defs': [],
                        'uses': [],
                        'is_state': isinstance(var, StateVariable)
                    }
                var_info[var_name]['defs'].append(node.node_id)
        
        # 转换为 DataFlowInfo
        for var_name, info in var_info.items():
            data_flow.append(DataFlowInfo(
                variable=var_name,
                definition_nodes=info['defs'],
                use_nodes=info['uses'],
                is_state_var=info['is_state'],
                depends_on=[]  # 简化版，完整实现需要深度分析
            ))
        
        return data_flow
    
    def _identify_state_access(self, func: Function) -> Tuple[Set[str], Set[str]]:
        """识别状态变量的读写"""
        reads = set()
        writes = set()
        
        for node in func.nodes:
            for var in node.state_variables_read:
                if var is not None and hasattr(var, 'name'):
                    reads.add(var.name)
            for var in node.state_variables_written:
                if var is not None and hasattr(var, 'name'):
                    writes.add(var.name)
        
        return reads, writes
    
    def _extract_internal_calls(self, func: Function) -> List[str]:
        """提取内部函数调用"""
        internal_calls = []
        
        for node in func.nodes:
            for ir in node.irs:
                if isinstance(ir, InternalCall):
                    if ir.function and hasattr(ir.function, 'name'):
                        internal_calls.append(ir.function.name)
        
        return list(set(internal_calls))  # 去重
    
    def save_to_jsonl(self, ckb: ContractKnowledgeBase, output_file: str):
        """保存知识库到 JSONL 文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # 写入类型信息
            type_info_dict = asdict(ckb.type_info)
            f.write(json.dumps({"type": "type_info", "data": type_info_dict}, ensure_ascii=False) + '\n')
            
            # 写入每个函数的知识
            for func_knowledge in ckb.functions:
                func_dict = asdict(func_knowledge)
                f.write(json.dumps({"type": "function", "data": func_dict}, ensure_ascii=False) + '\n')
        
        logger.info(f"[CKE] 知识库已保存到: {output_path}")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Context Knowledge Extraction - 提取智能合约知识库')
    parser.add_argument('sol_file', help='Solidity 源文件路径')
    parser.add_argument('-o', '--output', help='输出 JSONL 文件路径', 
                        default='cache/ckb.jsonl')
    parser.add_argument('--solc', help='指定 solc 版本', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    if not SLITHER_AVAILABLE:
        print("❌ Slither 未安装，请运行: pip install slither-analyzer")
        return 1
    
    try:
        # 提取知识
        extractor = SlitherCKExtractor(args.sol_file, args.solc)
        ckb = extractor.extract()
        
        # 保存到文件
        extractor.save_to_jsonl(ckb, args.output)
        
        # 打印摘要
        print(f"\n✅ 提取完成!")
        print(f"   合约: {ckb.type_info.contract_name}")
        print(f"   函数数量: {len(ckb.functions)}")
        
        sink_count = sum(len(f.sink_nodes) for f in ckb.functions)
        print(f"   Sink 节点总数: {sink_count}")
        
        return 0
    
    except Exception as e:
        logger.error(f"提取失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
