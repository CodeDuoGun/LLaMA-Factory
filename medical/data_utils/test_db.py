#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 audio_lable 表查询"""

from medical.data_utils.db import get_all_audio_labels, count_audio_labels, check_database_connection, fix_audio_file_prefix

if __name__ == "__main__":
    # 检查数据库连接
    if not check_database_connection():
        print("数据库连接失败，请检查网络和配置")
        exit(1)
        
    # 修复缺失前缀的 audio_file
    # print("\n=== 修复 audio_file 前缀 ===")
    # updated = fix_audio_file_prefix(start_id=1305)
    # print(f"已修复 {updated} 条记录")

    # 查询全部
    print("=== 查询全部记录 ===")
    all_records = get_all_audio_labels()
    print(f"表中共有 {len(all_records)} 条记录\n")

    # 查询前 5 条
    print("=== 查询前 5 条 ===")
    records = get_all_audio_labels(limit=5)
    for r in records:
        print(f"id={r.id}  audio_file={r.audio_file}  status={r.status}")

    # 查询最后 100 条
    total = count_audio_labels()
    offset = max(0, total - 100)
    print(f"\n=== 查询最后 100 条 (offset={offset}) ===")
    records = get_all_audio_labels(limit=100, offset=offset)
    for r in records:
        print(f"id={r.id}  audio_file={r.audio_file}  status={r.status}")



    print("\n测试完成")
