#!/usr/bin/env python3
"""批量执行智能合约审计"""

import os
import subprocess
import csv
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path('/home/zhuwei/PycharmProjects/MyLLM')
BENCHMARK_DIR = PROJECT_ROOT / 'evaluation' / 'benchmark' / 'contracts'

# 配置设置
CONFIG_NAME = "SmartContractCKD"  # 使用 myConfig 配置

# 注意: OPENAI_API_KEY 需要在系统环境变量中配置


def audit_contract(sol_file: Path, config: str = "myConfig"):
    """审计单个合约"""
    # 读取合约内容
    with open(sol_file, 'r', encoding='utf-8') as f:
        contract_content = f.read()

    # 检查是否设置了实验配置（从环境变量获取实验ID）
    experiment_id = os.environ.get('EXPERIMENT_ID', '')
    
    # 生成项目名称（使用相对于BENCHMARK_DIR的路径作为标识）
    relative_to_benchmark = sol_file.relative_to(BENCHMARK_DIR)
    # 将路径中的 / 替换为 _ 作为项目名
    path_parts = str(relative_to_benchmark.parent).replace('/', '_').replace('.', '')
    filename = sol_file.stem
    
    # 如果有实验ID，添加到项目名称前缀
    if experiment_id:
        project_name = f"{experiment_id}_{path_parts}_{filename}" if path_parts else f"{experiment_id}_{filename}"
    else:
        project_name = f"ContractAudit_{path_parts}_{filename}" if path_parts else f"ContractAudit_{filename}"

    # 使用绝对路径传递给 --sol 参数（确保代码能正确找到文件）
    absolute_sol_path = sol_file.resolve()

    # 构建命令 (--sol 参数使用绝对路径)
    command = [
        "python", "run.py",
        "--org", config,
        "--config", config,
        "--task", contract_content,
        "--name", project_name,
        "--sol", str(absolute_sol_path),  # 绝对路径，确保能正确定位文件
        "--model", "GPT_4_O_MINI"  # 使用 Claude Sonnet 4.5 避免 Gemini API 错误
    ]

    print(f"🔍 Auditing: {sol_file.relative_to(PROJECT_ROOT)}")
    print(f"   Config: {config}")
    print(f"   Experiment: {experiment_id if experiment_id else 'None'}")
    print(f"   Project: {project_name}")
    print(f"   Sol Path: {absolute_sol_path}")
    print("-" * 80)

    try:
        # 捕获输出，不显示在终端
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,  # 捕获输出，不显示在终端
            text=True,
            timeout=None  # 无超时限制
        )
        print("-" * 80)
        print(f"✅ Success: {project_name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 80)
        print(f"❌ Error: {project_name}")
        print(f"   Exit Code: {e.returncode}")
        print(f"   Error Output (last 1000 chars):")
        if e.stderr:
            print(f"   STDERR: {e.stderr[-1000:]}")
        if e.stdout:
            print(f"   STDOUT: {e.stdout[-1000:]}")
        print()
        return False


def collect_results_to_csv(output_csv: Path):
    """收集所有审计结果到统一的CSV文件"""
    warehouse_dir = PROJECT_ROOT / 'WareHouse'
    all_results = []
    
    # 遍历所有项目目录
    for project_dir in warehouse_dir.iterdir():
        if not project_dir.is_dir():
            continue
        
        # 查找binary_classification_result.csv
        csv_file = project_dir / 'binary_classification_result.csv'
        if csv_file.exists():
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_results.append({
                            'File Name': row['File Name'],
                            'Has Vulnerability': row['Has Vulnerability']
                        })
            except Exception as e:
                print(f"⚠️  Error reading {csv_file}: {e}")
    
    # 写入汇总CSV
    if all_results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['File Name', 'Has Vulnerability'])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n{'=' * 80}")
        print(f"📄 Results exported to: {output_csv}")
        print(f"   Total records: {len(all_results)}")
        print(f"{'=' * 80}")
    else:
        print("\n⚠️  No results found to export.")


def main():
    """主函数"""
    print("=" * 80)
    print(f"🚀 智能合约批量审计系统")
    print(f"📋 配置: {CONFIG_NAME}")
    print(f"📂 数据集: {BENCHMARK_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 80)
    
    # 统计
    total = 0
    success = 0
    failed = 0

    # 递归遍历所有.sol文件（包括子目录）
    all_sol_files = list(BENCHMARK_DIR.rglob('*.sol'))
    
    print(f"\n📊 Found {len(all_sol_files)} .sol files in total")
    print("=" * 80)

    for sol_file in sorted(all_sol_files):
        # 获取相对路径用于显示
        relative_path = sol_file.relative_to(BENCHMARK_DIR)
        
        print(f"\n{'=' * 60}")
        print(f"📁 Processing: {relative_path}")
        print(f"{'=' * 60}")
        
        total += 1
        # 显式使用 CONFIG_NAME
        if audit_contract(sol_file, config=CONFIG_NAME):
            success += 1
        else:
            failed += 1

    # 输出统计
    print(f"\n{'=' * 60}")
    print(f"📊 Summary")
    print(f"{'=' * 60}")
    print(f"Total:   {total}")
    print(f"Success: {success} ({success / total * 100:.1f}%)")
    print(f"Failed:  {failed} ({failed / total * 100:.1f}%)")
    
    # 汇总所有结果到CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = PROJECT_ROOT / 'results' / f'vulnerability_detection_results_{timestamp}.csv'
    output_csv.parent.mkdir(exist_ok=True)
    collect_results_to_csv(output_csv)


if __name__ == "__main__":
    main()