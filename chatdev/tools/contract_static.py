"""Utility helpers for collecting smart-contract intelligence via external analyzers.

This module integrates Mythril and Slither to extract:
- Transaction/trace skeletons (Mythril)
- Function-level control-flow logic (Slither CFG .dot)
- Raw Solidity source code

It exposes `run_contract_intel` which returns these artifacts in a single dict so
that phases such as ContractAnalysis can easily enrich their prompts.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import re
from pathlib import Path
from typing import Dict, Tuple

DEFAULT_TIMEOUT = 120  # seconds


class ContractIntelError(RuntimeError):
    """Raised when an external analyzer fails."""


def _detect_solidity_version(sol_file: Path) -> tuple:
    """从 Solidity 文件中检测 pragma 版本
    
    Args:
        sol_file: Solidity 源文件路径
        
    Returns:
        (完整版本, 主次版本, 版本前缀) 例如: ("0.4.16", "0.4", "^")
        版本前缀: ^ (兼容), >= (大于等于), = (精确), None (无前缀)
    """
    import re
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        content = sol_file.read_text(encoding='utf-8', errors='ignore')
        
        # 匹配 pragma solidity 语句
        # 支持格式: ^0.5.0, >=0.4.16 <0.6.0, 0.8.0, etc.
        match = re.search(r'pragma\s+solidity\s+([^;]+);', content, re.IGNORECASE)
        
        if match:
            version_str = match.group(1).strip()
            logger.debug(f"[Version] Found pragma: {version_str}")
            
            # 提取版本前缀和完整版本号
            # 例如: ^0.4.16 -> prefix=^, version=0.4.16
            prefix_match = re.search(r'([>=^]*)(\d+\.\d+\.\d+)', version_str)
            if prefix_match:
                prefix = prefix_match.group(1) or None
                full_version = prefix_match.group(2)
                major_minor = '.'.join(full_version.split('.')[:2])
                
                logger.info(f"[Version] Detected Solidity {prefix or ''}{full_version} for {sol_file.name}")
                return (full_version, major_minor, prefix)
            
            # 如果没有补丁版本号，只有主次版本 (例如 ^0.4)
            simple_match = re.search(r'([>=^]*)(\d+\.\d+)', version_str)
            if simple_match:
                prefix = simple_match.group(1) or None
                major_minor = simple_match.group(2)
                full_version = f"{major_minor}.0"
                
                logger.info(f"[Version] Detected Solidity {prefix or ''}{major_minor} for {sol_file.name}")
                return (full_version, major_minor, prefix)
        
        logger.warning(f"[Version] No pragma found in {sol_file.name}, using default")
        
    except Exception as e:
        logger.warning(f"[Version] Failed to detect version: {e}")
    
    # 默认版本
    return ("0.8.0", "0.8", None)


def _get_available_solc_versions() -> set:
    """获取系统中已安装的 solc 版本（完整三位版本号）
    
    Returns:
        已安装版本的集合，例如 {"0.4.25", "0.5.0", "0.8.0"}
    """
    import subprocess
    import logging
    
    logger = logging.getLogger(__name__)
    versions = set()
    
    try:
        # 尝试使用 solc-select 列出版本
        result = subprocess.run(
            ["solc-select", "versions"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # 解析输出，查找已安装的版本
            for line in result.stdout.split('\n'):
                # 格式: 0.5.0 (current, set by /...)
                match = re.search(r'(\d+\.\d+\.\d+)', line)
                if match:
                    version = match.group(1)
                    versions.add(version)  # 保留完整版本号
            
            logger.debug(f"[Version] Available solc versions: {sorted(versions)}")
    
    except FileNotFoundError:
        logger.warning("[Version] solc-select not found, version auto-selection disabled")
    except Exception as e:
        logger.warning(f"[Version] Failed to get solc versions: {e}")
    
    return versions


def _select_best_solc_version(required_full: str, required_major_minor: str, 
                              prefix: str, available_versions: set) -> str:
    """选择最佳的 solc 版本，支持语义化版本和向下兼容
    
    Args:
        required_full: 完整版本，例如 "0.4.16"
        required_major_minor: 主次版本，例如 "0.4"
        prefix: 版本前缀，例如 "^" (兼容), ">=" (大于等于), None (精确)
        available_versions: 可用版本集合，例如 {"0.4.15", "0.4.25", "0.5.0"}
        
    Returns:
        最佳匹配的版本，如果没有匹配则返回 None
        
    版本兼容性规则:
        - ^ (caret): 兼容同一主次版本，例如 ^0.4.16 可以用 0.4.16-0.4.x
        - >= (greater): 大于等于指定版本
        - 无前缀: 精确匹配
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not available_versions:
        return None
    
    def parse_version(v: str) -> tuple:
        """解析版本号为元组，例如 "0.4.16" -> (0, 4, 16)"""
        parts = v.split('.')
        return tuple(int(p) for p in parts)
    
    required_tuple = parse_version(required_full)
    
    # 1. 精确匹配
    if required_full in available_versions:
        logger.info(f"[Version] Exact match found: {required_full}")
        return required_full
    
    # 2. 根据前缀选择兼容版本
    if prefix == '^':
        # ^ 表示兼容同一主次版本，补丁版本可以更高
        # 例如: ^0.4.16 可以用 0.4.16, 0.4.17, ..., 0.4.26，但不能用 0.4.15 或 0.5.0
        compatible = []
        for v in available_versions:
            v_tuple = parse_version(v)
            # 同一主次版本 (0.4.x)
            if v_tuple[:2] == required_tuple[:2]:
                # 补丁版本 >= 要求的版本
                if v_tuple[2] >= required_tuple[2]:
                    compatible.append(v)
        
        if compatible:
            # 选择最接近的版本（最小的满足条件的版本）
            best = min(compatible, key=parse_version)
            logger.info(f"[Version] Compatible version (^) found: {best} for ^{required_full}")
            return best
        else:
            logger.warning(f"[Version] No compatible version for ^{required_full}")
            logger.warning(f"[Version] Available: {sorted(available_versions)}")
            logger.warning(f"[Version] Need: >= {required_full} and < {required_major_minor}.x")
    
    elif prefix == '>=':
        # >= 表示大于等于指定版本
        compatible = []
        for v in available_versions:
            v_tuple = parse_version(v)
            if v_tuple >= required_tuple:
                compatible.append(v)
        
        if compatible:
            # 选择最接近的版本（最小的满足条件的版本）
            best = min(compatible, key=parse_version)
            logger.info(f"[Version] Compatible version (>=) found: {best} for >={required_full}")
            return best
        else:
            logger.warning(f"[Version] No version >= {required_full}")
    
    else:
        # 无前缀，尝试同一主次版本的最高版本
        major_minor_versions = [v for v in available_versions 
                                if v.startswith(required_major_minor + '.')]
        
        if major_minor_versions:
            # 选择同一主次版本的最高版本
            best = max(major_minor_versions, key=parse_version)
            logger.info(f"[Version] Same major.minor version found: {best} for {required_full}")
            return best
        else:
            logger.warning(f"[Version] No version in {required_major_minor}.x series")
    
    return None


def _switch_solc_version(version: str) -> bool:
    """切换到指定的 solc 版本
    
    Args:
        version: 目标版本，例如 "0.5.0"
        
    Returns:
        是否成功切换
    """
    import subprocess
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # 使用 solc-select 切换版本
        result = subprocess.run(
            ["solc-select", "use", version],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info(f"[Version] Switched to solc {version}")
            return True
        else:
            logger.warning(f"[Version] Failed to switch to {version}: {result.stderr}")
            return False
    
    except FileNotFoundError:
        logger.warning("[Version] solc-select not found")
        return False
    except Exception as e:
        logger.warning(f"[Version] Failed to switch version: {e}")
        return False


def _ensure_solidity_file(sol_path: str | os.PathLike) -> Path:
    path = Path(sol_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Solidity file not found: {path}")
    if path.suffix.lower() != ".sol":
        raise ValueError(f"Expected a .sol file, got: {path}")
    return path


def _read_source(sol_path: Path) -> str:
    return sol_path.read_text(encoding="utf-8", errors="ignore")


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    else:
        proc.terminate()


def _run_external_command(cmd: Tuple[str, ...], timeout: int = DEFAULT_TIMEOUT) -> Tuple[int, str, str]:
    """Execute command with timeout, returning (returncode, stdout, stderr)."""
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    preexec_fn = os.setsid if os.name != "nt" else None
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
        preexec_fn=preexec_fn,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _terminate_process(proc)
        stdout, stderr = proc.communicate()
        raise ContractIntelError(f"Command timed out: {' '.join(cmd)}") from None
    return proc.returncode, stdout, stderr


def _extract_mythril_trace(raw_output: str, raw_error: str) -> str:
    """提取Mythril输出，优先使用stdout，其次stderr"""
    text = raw_output.strip()
    if not text:
        text = raw_error.strip()
    return text or "Mythril did not return any execution trace."


def _parse_mythril_vulnerabilities(raw_output: str) -> Dict[str, any]:
    """将Mythril输出解析为结构化数据"""
    from collections import Counter
    
    vulnerabilities = []
    
    # 按漏洞分割 (==== 标记)
    vuln_blocks = re.split(r'====\s+(.+?)\s+====', raw_output)
    
    for i in range(1, len(vuln_blocks), 2):
        vuln_name = vuln_blocks[i].strip()
        vuln_content = vuln_blocks[i+1] if i+1 < len(vuln_blocks) else ""
        
        # 提取关键信息
        swc_match = re.search(r'SWC ID:\s*(\d+)', vuln_content)
        severity_match = re.search(r'Severity:\s*(\w+)', vuln_content)
        function_match = re.search(r'Function name:\s*(.+)', vuln_content)
        contract_match = re.search(r'Contract:\s*(\w+)', vuln_content)
        
        vuln_info = {
            'name': vuln_name,
            'swc_id': swc_match.group(1) if swc_match else None,
            'severity': severity_match.group(1) if severity_match else 'Unknown',
            'function': function_match.group(1).strip() if function_match else None,
            'contract': contract_match.group(1) if contract_match else None,
            'description': vuln_content.strip()
        }
        vulnerabilities.append(vuln_info)
    
    severity_counts = dict(Counter(v['severity'] for v in vulnerabilities))
    
    return {
        'total_count': len(vulnerabilities),
        'vulnerabilities': vulnerabilities,
        'severity_summary': severity_counts
    }


def _format_vulnerabilities(vulnerabilities: list) -> str:
    """将漏洞列表格式化为可读文本"""
    if not vulnerabilities:
        return "No vulnerabilities detected."
    
    formatted = []
    for i, vuln in enumerate(vulnerabilities, 1):
        desc_preview = vuln['description'][:300].replace('\n', ' ')
        formatted.append(
            f"{i}. {vuln['name']}\n"
            f"   - SWC ID: {vuln['swc_id'] or 'N/A'}\n"
            f"   - Severity: {vuln['severity']}\n"
            f"   - Function: {vuln['function'] or 'N/A'}\n"
            f"   - Contract: {vuln['contract'] or 'N/A'}\n"
            f"   - Preview: {desc_preview}...\n"
        )
    return "\n".join(formatted)


def _extract_slither_logic(raw_output: str, raw_error: str) -> str:
    text = raw_output.strip()
    if not text:
        text = raw_error.strip()
    
    # 如果输出只包含文件路径（很少的行且包含.dot），返回更有意义的消息
    if text and text.count('\n') < 3 and '.dot' in text:
        return "Slither call-graph generated, but content not available. Check file: " + text
    
    return text or "Slither did not produce CFG information."


def _run_mythril(sol_file: Path, timeout: int) -> str:
    """运行Mythril分析，正确处理漏洞检测结果"""
    cmd = (
        "myth",
        "analyze",
        str(sol_file),
    )
    code, stdout, stderr = _run_external_command(cmd, timeout)
    
    # Mythril在检测到漏洞时会返回非0退出码，这是正常行为
    combined = stdout.strip() or stderr.strip()
    
    # 检查是否包含有效的漏洞报告标记
    if "====" in combined or "SWC ID" in combined:
        return combined
    
    # 真正的错误情况：没有输出且返回非0
    if code != 0 and not combined:
        raise ContractIntelError(
            f"Mythril failed with code {code}: No output generated"
        )
    
    return combined or "No vulnerabilities detected by Mythril."


def _run_slither(sol_file: Path, timeout: int) -> str:
    """运行Slither call-graph分析"""
    import re
    import logging
    
    logger = logging.getLogger(__name__)
    
    cmd = (
        "slither",
        str(sol_file),
        "--print",
        "call-graph",
    )
    code, stdout, stderr = _run_external_command(cmd, timeout)
    if code != 0:
        raise ContractIntelError(
            f"Slither failed with code {code}: {stderr.strip() or stdout.strip()}"
        )
    
    combined_output = stdout + "\n" + stderr
    
    # 使用正则表达式提取 .dot 文件路径
    pattern = r'([^\s]+\.call-graph\.dot)'
    matches = re.findall(pattern, combined_output)
    
    if matches:
        dot_path = matches[-1].strip('"').strip("'")
        dot_file = Path(dot_path)
        
        logger.info(f"[Slither] Found call-graph file: {dot_file.name}")
        
        if dot_file.exists():
            try:
                content = dot_file.read_text(encoding="utf-8", errors="ignore")
                logger.info(f"[Slither] Successfully read {len(content)} bytes from call-graph")
                return content
            except OSError as e:
                logger.warning(f"[Slither] Failed to read call-graph file: {e}")
        else:
            logger.warning(f"[Slither] Call-graph file not found at: {dot_file}")
    else:
        logger.warning("[Slither] No call-graph file path found in output")
    
    # 回退到原始输出
    logger.info("[Slither] Using raw output as fallback")
    return _extract_slither_logic(stdout, stderr)


def _clean_slither_output(raw_output: str) -> str:
    """清理Slither输出，移除调试信息"""
    lines = raw_output.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # 跳过solc命令行
        if stripped.startswith("'solc"):
            continue
        
        # 跳过所有INFO:开头的日志行（包括INFO:Printers:, INFO:Slither:等）
        if stripped.startswith("INFO:"):
            continue
        
        # 跳过running提示
        if "running" in line and "'solc" in line:
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def _extract_cfg_content(slither_output: str, sol_file: Path) -> str:
    """从Slither输出中提取并读取CFG DOT文件内容"""
    cfg_contents = []
    
    # 查找所有导出的DOT文件路径
    for line in slither_output.split('\n'):
        if 'Export' in line and '.dot' in line:
            # 提取文件路径
            match = re.search(r'Export\s+(.+\.dot)', line)
            if match:
                dot_path_str = match.group(1).strip()
                dot_path = Path(dot_path_str)
                
                # 尝试读取文件
                if dot_path.exists():
                    try:
                        content = dot_path.read_text(encoding='utf-8', errors='ignore')
                        # 提取函数名
                        function_name = dot_path.stem.split('-')[-1]
                        cfg_contents.append(
                            f"=== CFG for {function_name} ===\n{content}"
                        )
                    except Exception as e:
                        cfg_contents.append(f"Failed to read {dot_path.name}: {e}")
    
    if cfg_contents:
        return '\n\n'.join(cfg_contents)
    
    # 如果没有找到DOT文件，返回清理后的原始输出
    return slither_output


def _format_solidity_source(source_code: str) -> str:
    """格式化Solidity源代码，提高可读性
    
    当前策略：禁用格式化，直接返回原始代码。
    原因：简单的正则表达式无法正确处理复杂的Solidity语法（嵌套结构、多层大括号等），
         可能产生格式错误，影响代码可读性和行号准确性。
    
    未来改进：考虑集成专业的Solidity格式化工具（如prettier-plugin-solidity）。
    """
    # 禁用格式化，保持原样
    # 这样可以确保代码的完整性和行号的准确性
    return source_code


def _add_line_numbers(source_code: str) -> str:
    """为源代码添加行号"""
    lines = source_code.split('\n')
    numbered_lines = []
    for i, line in enumerate(lines, 1):
        numbered_lines.append(f"{i:4d} | {line}")
    return '\n'.join(numbered_lines)


def _run_slither_detector(sol_file: Path, detector: str, timeout: int) -> str:
    """运行特定的Slither检测器或打印器，支持自动版本检测和切换"""
    import re
    import logging
    
    logger = logging.getLogger(__name__)
    
    # 🔧 自动版本检测和切换
    full_version, major_minor, prefix = _detect_solidity_version(sol_file)
    available_versions = _get_available_solc_versions()
    
    if available_versions:
        best_version = _select_best_solc_version(full_version, major_minor, prefix, available_versions)
        if best_version:
            _switch_solc_version(best_version)
        else:
            logger.warning(f"[Version] No suitable solc version for {prefix or ''}{full_version}, using current")
    
    cmd = (
        "slither",
        str(sol_file),
        "--print",
        detector,
    )
    try:
        code, stdout, stderr = _run_external_command(cmd, timeout)
        combined = (stdout + "\n" + stderr).strip()
        
        # 检查是否是编译错误
        if "InvalidCompilation" in combined or "Error: Expected identifier" in combined:
            logger.error(f"[Slither] Compilation failed for {sol_file.name}")
            logger.error(f"[Slither] Required version: {prefix or ''}{full_version}, Available: {sorted(available_versions)}")
            return f"Slither compilation failed (Solidity {prefix or ''}{full_version} incompatibility). Please install: pip install solc-select && solc-select install {full_version}"
        
        # 清理输出
        cleaned = _clean_slither_output(combined)
        
        # 如果是CFG，尝试读取DOT文件内容
        if detector == 'cfg' and cleaned:
            return _extract_cfg_content(cleaned, sol_file)
        
        # 🔧 新增：如果是 call-graph，尝试读取 .dot 文件内容
        if detector == 'call-graph':
            # 使用正则表达式提取 .dot 文件路径
            pattern = r'([^\s]+\.call-graph\.dot)'
            matches = re.findall(pattern, combined)
            
            if matches:
                dot_path = matches[-1].strip('"').strip("'")
                dot_file = Path(dot_path)
                
                logger.info(f"[Slither] Found call-graph file: {dot_file.name}")
                
                if dot_file.exists():
                    try:
                        content = dot_file.read_text(encoding="utf-8", errors="ignore")
                        logger.info(f"[Slither] Successfully read {len(content)} bytes from call-graph")
                        return content
                    except OSError as e:
                        logger.warning(f"[Slither] Failed to read call-graph file: {e}")
                else:
                    logger.warning(f"[Slither] Call-graph file not found at: {dot_file}")
            else:
                logger.warning("[Slither] No call-graph file path found in output")
        
        return cleaned or f"No output from Slither {detector}"
    except Exception as e:
        return f"Slither {detector} unavailable: {e}"


def _run_slither_enhanced(sol_file: Path, timeout: int) -> Dict[str, str]:
    """运行多个Slither检测器获取全面信息"""
    results = {}
    
    # 1. 控制流图 (CFG)
    results['cfg'] = _run_slither_detector(sol_file, 'cfg', timeout)
    
    # 2. 函数摘要
    results['function_summary'] = _run_slither_detector(sol_file, 'function-summary', timeout)
    
    # 3. 调用图 (保持向后兼容)
    results['call_graph'] = _run_slither_detector(sol_file, 'call-graph', timeout)
    
    # 4. 人类可读的摘要
    results['human_summary'] = _run_slither_detector(sol_file, 'human-summary', timeout)
    
    return results


def run_contract_intel(
    sol_path: str | os.PathLike,
    *,
    mythril_timeout: int = DEFAULT_TIMEOUT,
    slither_timeout: int = DEFAULT_TIMEOUT,
    enhanced: bool = True,
) -> Dict[str, str]:
    """Run Mythril + Slither and return their artifacts along with raw source.
    
    Args:
        sol_path: Path to Solidity file
        mythril_timeout: Timeout for Mythril analysis
        slither_timeout: Timeout for Slither analysis
        enhanced: If True, run enhanced Slither analysis with multiple detectors
    
    Returns:
        Dictionary containing analysis results and metadata
    """
    sol_file = _ensure_solidity_file(sol_path)
    source_code = _read_source(sol_file)
    
    # 格式化源代码（当前禁用，直接返回原始代码）
    formatted_source = _format_solidity_source(source_code)
    # 为源代码添加行号
    source_with_line_numbers = _add_line_numbers(formatted_source)

    mythril_trace = ""
    mythril_error = None
    mythril_parsed = {'total_count': 0, 'vulnerabilities': [], 'severity_summary': {}}
    
    try:
        mythril_trace = _run_mythril(sol_file, mythril_timeout)
        # 解析Mythril输出为结构化数据
        mythril_parsed = _parse_mythril_vulnerabilities(mythril_trace)
    except (FileNotFoundError, ContractIntelError) as exc:
        mythril_error = f"Mythril unavailable: {exc}"

    slither_logic = ""
    slither_error = None
    slither_enhanced_data = {}
    
    try:
        if enhanced:
            slither_enhanced_data = _run_slither_enhanced(sol_file, slither_timeout)
            # 保持向后兼容，使用call-graph作为默认logic
            slither_logic = slither_enhanced_data.get('call_graph', '')
        else:
            slither_logic = _run_slither(sol_file, slither_timeout)
    except (FileNotFoundError, ContractIntelError) as exc:
        slither_error = f"Slither unavailable: {exc}"

    payload = {
        # 原始输出 (向后兼容)
        "mythril_trace": mythril_trace or (mythril_error or ""),
        "slither_logic": slither_logic or (slither_error or ""),
        "source_code": formatted_source,  # 原始源代码（格式化已禁用）
        
        # 新增：带行号的源代码
        "source_code_with_line_numbers": source_with_line_numbers,
        
        # 结构化Mythril数据
        "mythril_vuln_count": mythril_parsed['total_count'],
        "mythril_severity_summary": str(mythril_parsed['severity_summary']),
        "mythril_structured_report": _format_vulnerabilities(mythril_parsed['vulnerabilities']),
        "mythril_vulnerabilities": mythril_parsed['vulnerabilities'],  # 原始列表
        
        # 增强的Slither数据
        "slither_cfg": slither_enhanced_data.get('cfg', ''),
        "slither_function_summary": slither_enhanced_data.get('function_summary', ''),
        "slither_human_summary": slither_enhanced_data.get('human_summary', ''),
    }
    return payload


__all__ = ["run_contract_intel", "ContractIntelError"]
