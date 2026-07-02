#!/usr/bin/env python3
"""将 wuweiping_template_prescription.json 转换为 txt 格式"""

import json

def convert_json_to_txt(json_path: str, txt_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("中医方剂模板列表\n")
        f.write("=" * 80 + "\n\n")

        for template in data.get('templates', []):
            f.write("-" * 80 + "\n")
            f.write(f"模板ID: {template.get('template_id', 'N/A')}\n")
            f.write(f"方剂名称: {template.get('template_name', 'N/A')}\n")
            f.write("-" * 80 + "\n")

            drugs = template.get('drugs', [])
            if drugs:
                f.write("【药物组成】\n")
                for i, drug in enumerate(drugs, 1):
                    drug_name = drug.get('drug_name', 'N/A')
                    quantity = drug.get('quantity', 'N/A')
                    unit = drug.get('unit', '')
                    drug_form = drug.get('drug_form', '')
                    f.write(f"  {i}. {drug_name}")
                    if quantity != 'N/A':
                        f.write(f"  {quantity}{unit}")
                    if drug_form:
                        f.write(f"  ({drug_form})")
                    f.write("\n")
            else:
                f.write("【药物组成】无\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write(f"共计 {len(data.get('templates', []))} 个方剂模板\n")
        f.write("=" * 80 + "\n")

    print(f"转换完成！已保存到: {txt_path}")

if __name__ == "__main__":
    json_path = "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/wuweiping_template_prescription.json"
    txt_path = "/Users/tangxueduo/Projects/LLaMA-Factory/medical/data/wuweiping_template_prescription.txt"
    convert_json_to_txt(json_path, txt_path)
