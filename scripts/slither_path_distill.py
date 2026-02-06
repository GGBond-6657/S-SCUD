#!/usr/bin/env python3
"""
Context Knowledge Distillation (CKD) Module
路径切片与蒸馏 - 提取最小可疑路径上下文

核心功能：
1. Path Selection: 从 Sink 节点后向切片，找到所有控制依赖
2. Dependent Method Resolution: 递归提取依赖函数的最简路径
3. Variable Distillation: 收集路径相关的状态变量定义
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class PathSlice:
    """路径切片（从 Sink 节点后向切片得到）"""
    slice_id: str
    function: str
    sink_node_id: int
    sink_type: str
    
    # 路径上的关键信息
    control_nodes: List[int]      # 控制该路径的 IF/REQUIRE 节点
    expressions: List[str]         # 节点表达式（按执行顺序）
    guards: List[str]              # 守卫条件（需满足的条件）
    
    # 依赖信息
    state_vars_read: List[str]
    state_vars_written: List[str]
    dependent_functions: List[str]  # 需要分析的依赖函数
    
    # 风险评分
    risk_score: float
    risk_factors: List[str]


@dataclass
class DistilledContext:
    """蒸馏后的上下文（准备送给 LLM）"""
    contract: str
    function_signature: str
    path_slices: List[PathSlice]
    
    # 相关定义（最小上下文）
    state_var_definitions: Dict[str, str]  # {var_name: type_definition}
    dependent_function_code: Dict[str, str]  # {func_name: source_code}
    
    # 元信息
    total_risk_score: float
    recommended_slices: List[str]  # 推荐检查的切片 ID
    
    # 函数元数据（新增）
    visibility: str = 'public'  # 可见性
    modifiers: List[str] = None  # 修饰符列表
    
    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []


class PathDistiller:
    """路径蒸馏器"""
    
    def __init__(self, ckb_file: str):
        """
        初始化蒸馏器
        
        Args:
            ckb_file: CKE 生成的 JSONL 知识库文件
        """
        self.ckb_file = Path(ckb_file)
        if not self.ckb_file.exists():
            raise FileNotFoundError(f"知识库文件不存在: {ckb_file}")
        
        # 加载知识库
        self.type_info = None
        self.functions = {}  # {func_name: FunctionKnowledge}
        self._load_ckb()
        
        logger.info(f"[CKD] 加载知识库: {len(self.functions)} 个函数")
    
    def _load_ckb(self):
        """加载 CKB JSONL 文件"""
        with open(self.ckb_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if entry['type'] == 'type_info':
                    self.type_info = entry['data']
                elif entry['type'] == 'function':
                    func_data = entry['data']
                    self.functions[func_data['function']] = func_data
    
    def distill(self, top_k: int = 5, risk_threshold: float = 0.0) -> List[DistilledContext]:
        """
        蒸馏所有函数，返回高风险路径的上下文
        
        Args:
            top_k: 每个函数最多返回 K 条路径
            risk_threshold: 风险阈值（低于此值的路径被过滤）
        
        Returns:
            蒸馏后的上下文列表
        """
        distilled_contexts = []
        
        for func_name, func_data in self.functions.items():
            if not func_data['sink_nodes']:
                logger.debug(f"[CKD] 跳过无 Sink 的函数: {func_name}")
                continue
            
            logger.info(f"[CKD] 蒸馏函数: {func_name} ({len(func_data['sink_nodes'])} 个 Sink)")
            
            # 对每个 Sink 节点进行后向切片
            path_slices = []
            for sink_id in func_data['sink_nodes']:
                path_slice = self._backward_slice(func_data, sink_id)
                
                # 计算风险评分
                path_slice.risk_score = self._calculate_risk(path_slice)
                logger.info(f"  路径 {path_slice.slice_id}: 风险评分={path_slice.risk_score:.1f}, Sink={path_slice.sink_type}, 守卫={len(path_slice.guards)}个")
                
                if path_slice.risk_score >= risk_threshold:
                    path_slices.append(path_slice)
                else:
                    logger.debug(f"  ❌ 路径被过滤（低于阈值{risk_threshold}）")
            
            # 按风险排序，取 Top-K
            path_slices.sort(key=lambda p: p.risk_score, reverse=True)
            top_slices = path_slices[:top_k]
            
            if not top_slices:
                continue
            
            # 收集依赖的状态变量和函数
            state_var_defs = self._collect_state_var_definitions(top_slices)
            dep_func_code = self._collect_dependent_functions(top_slices)
            
            # 提取函数元数据
            visibility = func_data.get('visibility', 'public')
            modifiers = func_data.get('modifiers', [])
            
            distilled_context = DistilledContext(
                contract=self.type_info['contract_name'],
                function_signature=func_data['signature'],
                path_slices=top_slices,
                state_var_definitions=state_var_defs,
                dependent_function_code=dep_func_code,
                total_risk_score=sum(s.risk_score for s in top_slices),
                recommended_slices=[s.slice_id for s in top_slices[:3]],  # 推荐前 3 条
                visibility=visibility,
                modifiers=modifiers
            )
            
            distilled_contexts.append(distilled_context)
        
        logger.info(f"[CKD] 蒸馏完成: {len(distilled_contexts)} 个函数有高风险路径")
        return distilled_contexts
    
    def _backward_slice(self, func_data: Dict, sink_id: int) -> PathSlice:
        """
        从 Sink 节点开始后向切片（基于前驱节点遍历）
        
        注意：尽管字段名为'dominators'，实际存储的是前驱节点(predecessors)
        这是在slither_ck_extractor.py中使用node.fathers填充的
        
        Args:
            func_data: 函数知识数据
            sink_id: Sink 节点 ID
        
        Returns:
            PathSlice 对象
        """
        # 构建节点映射
        nodes = {n['node_id']: n for n in func_data['cfg_nodes']}
        
        if sink_id not in nodes:
            raise ValueError(f"Sink 节点 {sink_id} 不存在")
        
        sink_node = nodes[sink_id]
        
        # 后向遍历，找到所有前驱节点（BFS backward slicing）
        visited = set()
        control_nodes = []
        expressions = []
        guards = []
        internal_calls_in_slice = set()  # 记录切片中的函数调用
        
        queue = deque([sink_id])
        visited.add(sink_id)
        
        while queue:
            node_id = queue.popleft()
            node = nodes[node_id]
            
            # 记录节点信息
            if node['expression']:
                expressions.append(node['expression'])
                
                # 检测函数调用（改进：只记录切片路径上的调用）
                for func_name in func_data['internal_calls']:
                    if func_name in node['expression']:
                        internal_calls_in_slice.add(func_name)
            
            # 识别控制节点（IF、REQUIRE）
            if 'IF' in node['node_type'] or 'require' in node['expression'].lower():
                control_nodes.append(node_id)
                guards.append(node['expression'])
            
            # 添加前驱节点（注意：字段名为dominators但实际是predecessors）
            for pred_id in node.get('dominators', []):
                if pred_id not in visited and pred_id in nodes:
                    visited.add(pred_id)
                    queue.append(pred_id)
        
        # 反转表达式顺序（从入口到 Sink）
        expressions.reverse()
        
        # 提取状态变量读写（改进：只包含切片路径上的数据流）
        state_reads = set()
        state_writes = set()
        for df in func_data['data_flow']:
            if df['is_state_var']:
                # 检查定义节点或使用节点是否在切片中
                has_def_in_slice = any(node_id in visited for node_id in df['definition_nodes'])
                has_use_in_slice = any(node_id in visited for node_id in df['use_nodes'])
                
                if has_use_in_slice:
                    state_reads.add(df['variable'])
                if has_def_in_slice:
                    state_writes.add(df['variable'])
        
        # 提取依赖函数（改进：只包含切片路径上调用的函数）
        dependent_funcs = []
        for func_name in internal_calls_in_slice:
            if func_name in self.functions:
                dependent_funcs.append(func_name)
        
        slice_id = f"{func_data['function']}_sink{sink_id}"
        
        return PathSlice(
            slice_id=slice_id,
            function=func_data['function'],
            sink_node_id=sink_id,
            sink_type=sink_node.get('sink_type', 'unknown'),
            control_nodes=control_nodes,
            expressions=expressions[:20],  # 限制长度
            guards=guards,
            state_vars_read=list(state_reads),
            state_vars_written=list(state_writes),
            dependent_functions=dependent_funcs,
            risk_score=0.0,  # 稍后计算
            risk_factors=[]
        )
    
    def _calculate_risk(self, path_slice: PathSlice) -> float:
        """
        计算路径风险评分（启发式）
        
        风险因素：
        1. Sink 类型（call.value, selfdestruct 等高危）
        2. 缺少守卫条件
        3. 状态变量在外部调用前被修改
        4. 复杂的依赖函数调用
        """
        score = 0.0
        risk_factors = []
        
        # 1. Sink 类型权重
        sink_weights = {
            'call.value': 10.0,
            'selfdestruct': 10.0,
            'delegatecall': 8.0,
            'external_call': 5.0,
            'unchecked_call': 7.0,
            'state_write': 3.0,
        }
        sink_score = sink_weights.get(path_slice.sink_type, 1.0)
        score += sink_score
        risk_factors.append(f"Sink类型: {path_slice.sink_type} (+{sink_score})")
        
        # 2. 守卫条件不足
        if len(path_slice.guards) == 0:
            score += 5.0
            risk_factors.append("缺少守卫条件 (+5.0)")
        elif len(path_slice.guards) < 2:
            score += 2.0
            risk_factors.append("守卫条件较少 (+2.0)")
        
        # 3. 状态变量写入（重入风险）
        if path_slice.state_vars_written:
            score += len(path_slice.state_vars_written) * 2.0
            risk_factors.append(f"状态写入 {len(path_slice.state_vars_written)} 个 (+{len(path_slice.state_vars_written)*2.0})")
        
        # 4. 依赖函数复杂度
        if len(path_slice.dependent_functions) > 3:
            score += 3.0
            risk_factors.append(f"依赖函数多 ({len(path_slice.dependent_functions)}) (+3.0)")
        
        path_slice.risk_factors = risk_factors
        return score
    
    def _collect_state_var_definitions(self, path_slices: List[PathSlice]) -> Dict[str, str]:
        """收集路径涉及的状态变量定义"""
        var_defs = {}
        
        all_vars = set()
        for path in path_slices:
            all_vars.update(path.state_vars_read)
            all_vars.update(path.state_vars_written)
        
        for var_name in all_vars:
            if var_name in self.type_info['state_vars']:
                var_info = self.type_info['state_vars'][var_name]
                var_defs[var_name] = f"{var_info['type']} {var_info['visibility']} {var_name}"
        
        return var_defs
    
    def _collect_dependent_functions(self, path_slices: List[PathSlice]) -> Dict[str, str]:
        """收集依赖函数的代码（简化版）"""
        func_code = {}
        
        all_deps = set()
        for path in path_slices:
            all_deps.update(path.dependent_functions)
        
        for func_name in all_deps:
            if func_name in self.functions:
                func_data = self.functions[func_name]
                # 简化：返回函数签名和表达式摘要
                expressions = []
                for node in func_data['cfg_nodes'][:10]:  # 限制前 10 个节点
                    if node['expression']:
                        expressions.append(node['expression'])
                
                func_code[func_name] = {
                    'signature': func_data['signature'],
                    'visibility': func_data['visibility'],
                    'modifiers': func_data['modifiers'],
                    'expressions': expressions
                }
        
        return func_code
    
    def save_to_json(self, contexts: List[DistilledContext], output_file: str):
        """保存蒸馏结果到 JSON 文件"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [asdict(ctx) for ctx in contexts]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[CKD] 蒸馏结果已保存到: {output_path}")
    
    def validate_slice(self, path_slice: PathSlice, func_data: Dict) -> Dict[str, any]:
        """
        验证切片的有效性和质量
        
        Returns:
            验证报告字典
        """
        report = {
            'slice_id': path_slice.slice_id,
            'is_valid': True,
            'warnings': [],
            'statistics': {}
        }
        
        # 统计信息
        report['statistics'] = {
            'total_nodes': len(path_slice.expressions),
            'control_nodes': len(path_slice.control_nodes),
            'guards': len(path_slice.guards),
            'state_reads': len(path_slice.state_vars_read),
            'state_writes': len(path_slice.state_vars_written),
            'dependent_functions': len(path_slice.dependent_functions)
        }
        
        # 检查1：切片是否为空
        if not path_slice.expressions:
            report['warnings'].append("切片为空：没有找到任何表达式")
            report['is_valid'] = False
        
        # 检查2：是否包含函数入口
        entry_keywords = ['ENTRY_POINT', 'BEGIN', 'function']
        has_entry = any(kw in expr for expr in path_slice.expressions for kw in entry_keywords)
        if not has_entry:
            report['warnings'].append("可能缺少函数入口节点")
        
        # 检查3：高风险但无守卫
        high_risk_sinks = ['call.value', 'selfdestruct', 'delegatecall']
        if path_slice.sink_type in high_risk_sinks and len(path_slice.guards) == 0:
            report['warnings'].append(f"高风险Sink({path_slice.sink_type})缺少守卫条件")
        
        # 检查4：切片大小合理性
        if len(path_slice.expressions) > 50:
            report['warnings'].append(f"切片过大({len(path_slice.expressions)}个节点)，可能包含过多噪音")
        
        # 检查5：依赖函数过多
        if len(path_slice.dependent_functions) > 10:
            report['warnings'].append(f"依赖函数过多({len(path_slice.dependent_functions)}个)，可能需要进一步过滤")
        
        return report


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Context Knowledge Distillation - 路径蒸馏')
    parser.add_argument('ckb_file', help='CKB JSONL 文件路径')
    parser.add_argument('-o', '--output', help='输出 JSON 文件路径',
                        default='cache/distilled.json')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                        help='每个函数最多保留 K 条路径')
    parser.add_argument('-t', '--threshold', type=float, default=5.0,
                        help='风险阈值（低于此值的路径被过滤）')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    try:
        # 蒸馏路径
        distiller = PathDistiller(args.ckb_file)
        contexts = distiller.distill(top_k=args.top_k, risk_threshold=args.threshold)
        
        # 保存结果
        distiller.save_to_json(contexts, args.output)
        
        # 打印摘要
        print(f"\n✅ 蒸馏完成!")
        print(f"   高风险函数: {len(contexts)}")
        
        total_slices = sum(len(ctx.path_slices) for ctx in contexts)
        print(f"   路径切片总数: {total_slices}")
        
        if contexts:
            top_ctx = max(contexts, key=lambda c: c.total_risk_score)
            print(f"   最高风险函数: {top_ctx.function_signature} (评分: {top_ctx.total_risk_score:.1f})")
        
        return 0
    
    except Exception as e:
        logger.error(f"蒸馏失败: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
