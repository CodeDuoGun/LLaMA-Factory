"""
模板方分析脚本
功能：
1. 调用LLM分析模板方内容
2. 识别不同类别模板方的适用疾病和证候、治则治法
3. 识别同类模板的核心药物
4. 分析加减药的规则
"""

import json
import os
import sys
import re
from typing import Optional
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical.llm_v2.doubao import DoubaoAIClient
from medical.config import config
from medical.utils.log import logger


class TemplatePrescriptionAnalyzer:
    """模板方分析器"""

    def __init__(self, llm_client: Optional[DoubaoAIClient] = None):
        """初始化分析器"""
        if llm_client is None:
            self.client = DoubaoAIClient(
                api_key=config.ARK_API_KEY,
                base_url=config.ARK_API_URL
            )
        else:
            self.client = llm_client

        self.model = config.DOUBAO_TEXT_MODEL

    def load_templates(self, filepath: str) -> list:
        """加载模板方数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('templates', [])

    def format_drugs(self, drugs: list) -> str:
        """格式化药物列表"""
        drug_list = []
        for drug in drugs:
            name = drug.get('drug_name', '')
            qty = drug.get('quantity', '')
            unit = drug.get('unit', 'g')
            drug_list.append(f"{name} {qty}{unit}")
        return "、".join(drug_list)

    def analyze_single_template(self, template: dict) -> dict:
        """
        分析单个模板方

        Returns:
            dict: 包含分析结果的字典
        """
        template_name = template.get('template_name', '')
        template_id = template.get('template_id', '')
        drugs = template.get('drugs', [])

        if not drugs:
            return {
                'template_id': template_id,
                'template_name': template_name,
                'error': 'No drugs found'
            }

        drugs_text = self.format_drugs(drugs)

        prompt = f"""你是一位经验丰富的中医专家。请分析以下模板方的药物组成，提取关键信息。

模板方名称：{template_name}
药物组成：{drugs_text}

请从以下维度进行分析：

1. **适用证候**（主要症状和体征）：根据药物组成推断该方主要治疗什么证候
2. **适用疾病**（现代医学病名或中医病名）：推断该方主要治疗哪些疾病
3. **治则治法**：该方的治疗原则和方法
4. **核心药物**：方中最重要的药物（通常出现频率高或剂量大，或起君药作用）
5. **加减药规则**：
   - 加药指征：什么情况下需要增加哪些药物
   - 减药指征：什么情况下需要减少哪些药物

请用JSON格式输出，结构如下：
{{
    "applicable_symptoms": ["症状1", "症状2"],
    "applicable_diseases": ["疾病1", "疾病2"],
    "treatment_principles": ["治法1", "治法2"],
    "core_drugs": ["药物1", "药物2", "药物3"],
    "addition_rules": [
        {{"condition": "加药指征", "add_drugs": ["药物名称"]}}
    ],
    "subtraction_rules": [
        {{"condition": "减药指征", "subtract_drugs": ["药物名称"]}}
    ],
    "analysis_note": "简要说明分析依据"
}}

只输出JSON，不要有其他内容。"""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat(
                messages=messages,
                llm_model=self.model,
                max_tokens=1500,
                temperature=0.3
            )

            # 尝试解析JSON
            result = self._parse_json_response(response)
            if result:
                result['template_id'] = template_id
                result['template_name'] = template_name
                return result
            else:
                return {
                    'template_id': template_id,
                    'template_name': template_name,
                    'raw_response': response,
                    'error': 'Failed to parse JSON'
                }

        except Exception as e:
            logger.error(f"Error analyzing template {template_name}: {e}")
            return {
                'template_id': template_id,
                'template_name': template_name,
                'error': str(e)
            }

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """解析LLM返回的JSON响应"""
        # 尝试直接解析
        try:
            return json.loads(response)
        except:
            pass

        # 尝试提取JSON块
        json_pattern = r'\{[\s\S]*\}'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                result = json.loads(match)
                if 'applicable_symptoms' in result or 'core_drugs' in result:
                    return result
            except:
                continue
        return None

    def group_templates_by_core_drugs(self, templates: list, analysis_results: list) -> dict:
        """
        根据核心药物对模板方进行分组

        Args:
            templates: 原始模板列表
            analysis_results: 分析结果列表

        Returns:
            dict: 分组后的结果
        """
        # 建立template_id到analysis的映射
        analysis_map = {r['template_id']: r for r in analysis_results}

        # 按核心药物分组
        groups = {}
        for template in templates:
            template_id = template['template_id']
            analysis = analysis_map.get(template_id, {})

            if 'error' in analysis:
                continue

            core_drugs = analysis.get('core_drugs', [])
            if not core_drugs:
                continue

            # 使用核心药物的组合作为key
            key = '|'.join(sorted(core_drugs[:3]))  # 使用前3个核心药物

            if key not in groups:
                groups[key] = {
                    'core_drugs': core_drugs,
                    'templates': []
                }

            groups[key]['templates'].append({
                'template_id': template_id,
                'template_name': template['template_name'],
                'drugs': self.format_drugs(template['drugs']),
                'applicable_symptoms': analysis.get('applicable_symptoms', []),
                'applicable_diseases': analysis.get('applicable_diseases', []),
                'treatment_principles': analysis.get('treatment_principles', []),
                'addition_rules': analysis.get('addition_rules', []),
                'subtraction_rules': analysis.get('subtraction_rules', []),
            })

        return groups

    def analyze_similar_template_rules(self, group: dict) -> dict:
        """
        分析同类模板方的加减药规则

        Args:
            group: 包含多个相似模板的分组

        Returns:
            dict: 总结的加减药规则
        """
        templates = group.get('templates', [])
        if len(templates) < 2:
            return {'message': '模板数量不足，无法进行对比分析'}

        prompt = f"""你是中医专家。请分析以下同类模板方的异同，提取加减药规则。

核心药物：{', '.join(group['core_drugs'])}

以下是{len(templates)}个模板方：

"""

        for i, t in enumerate(templates, 1):
            prompt += f"""
模板{i}: {t['template_name']}
药物组成：{t['drugs']}
适用证候：{', '.join(t['applicable_symptoms'])}
治则治法：{', '.join(t['treatment_principles'])}
"""

        prompt += """
请分析：
1. 这些模板方的共同点（相同药物、相似治法）
2. 每个模板的特色药物及其作用
3. 什么情况下应该选用哪个模板
4. 能否归纳出一套加减药规则

请用JSON格式输出：
{
    "common_features": {
        "common_drugs": ["共同药物"],
        "common_treatment_principles": ["共同治法"]
    },
    "template_variations": [
        {
            "template_name": "模板名",
            "special_drugs": ["特色药物"],
            "special_effects": ["特色作用"]
        }
    ],
    "selection_rules": ["选择规则"],
    "generalized_addition_subtraction_rules": {
        "addition": [{"condition": "加药条件", "add_drugs": ["药物"], "reason": "原因"}],
        "subtraction": [{"condition": "减药条件", "subtract_drugs": ["药物"], "reason": "原因"}]
    }
}

只输出JSON，不要有其他内容。"""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat(
                messages=messages,
                llm_model=self.model,
                max_tokens=2000,
                temperature=0.3
            )

            result = self._parse_json_response(response)
            return result or {'raw_response': response}

        except Exception as e:
            logger.error(f"Error analyzing group rules: {e}")
            return {'error': str(e)}

    def analyze_all_templates(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 5,
        delay: float = 1.0
    ) -> dict:
        """
        分析所有模板方

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            batch_size: 每批分析数量
            delay: 请求间隔（秒）

        Returns:
            dict: 分析结果摘要
        """
        import time

        templates = self.load_templates(input_file)
        logger.info(f"Loaded {len(templates)} templates")

        results = []
        total = len(templates)

        for i, template in enumerate(templates):
            logger.info(f"Analyzing template {i+1}/{total}: {template.get('template_name', '')}")

            result = self.analyze_single_template(template)
            results.append(result)

            # 每批保存一次
            if (i + 1) % batch_size == 0:
                self._save_intermediate_results(output_file, results)
                logger.info(f"Saved intermediate results ({i+1}/{total})")

            # 避免请求过快
            time.sleep(delay)

        # 最终保存
        self._save_intermediate_results(output_file, results)

        # 汇总统计
        success_count = sum(1 for r in results if 'error' not in r)
        summary = {
            'total_templates': total,
            'success_count': success_count,
            'failed_count': total - success_count,
            'results': results
        }

        # 保存完整报告
        report_file = output_file.replace('.json', '_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Analysis complete. Results saved to {output_file}")
        logger.info(f"Report saved to {report_file}")

        return summary

    def _save_intermediate_results(self, output_file: str, results: list):
        """保存中间结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'analyses': results}, f, ensure_ascii=False, indent=2)

    def generate_summary_report(self, analysis_file: str, output_file: str = None):
        """
        生成汇总报告，包括分组分析和加减药规则

        Args:
            analysis_file: 分析结果文件
            output_file: 输出文件路径
        """
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = data.get('analyses', [])
        templates = self.load_templates(
            analysis_file.replace('_analysis.json', '_report.json').replace('_report.json', '.json')
        )

        # 如果模板未加载，尝试加载原始文件
        if not templates:
            original_file = 'medical/data/wuweiping_template_prescription.json'
            if os.path.exists(original_file):
                templates = self.load_templates(original_file)

        # 按核心药物分组
        groups = self.group_templates_by_core_drugs(templates, results)

        # 对每个分组进行深入分析
        group_analysis = {}
        for key, group in groups.items():
            if len(group['templates']) >= 2:
                logger.info(f"Analyzing group with core drugs: {key[:50]}...")
                rules = self.analyze_similar_template_rules(group)
                group_analysis[key] = {
                    'core_drugs': group['core_drugs'],
                    'template_count': len(group['templates']),
                    'templates': group['templates'],
                    'rules': rules
                }

        # 生成完整报告
        report = {
            'summary': {
                'total_templates': len(results),
                'grouped_templates': sum(g['template_count'] for g in group_analysis.values()),
                'groups_count': len(group_analysis)
            },
            'groups': group_analysis,
            'ungrouped_templates': [
                {
                    'template_id': r['template_id'],
                    'template_name': r['template_name'],
                    'core_drugs': r.get('core_drugs', [])
                }
                for r in results
                if 'error' not in r and not r.get('core_drugs')
            ]
        }

        if output_file is None:
            output_file = analysis_file.replace('.json', '_summary.json')

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary report saved to {output_file}")
        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='模板方分析工具')
    parser.add_argument('--input', '-i', default='medical/data/wuweiping_template_prescription.json',
                        help='输入模板方文件')
    parser.add_argument('--output', '-o', default='medical/data/template_analysis_results.json',
                        help='输出分析结果文件')
    parser.add_argument('--batch-size', '-b', type=int, default=5,
                        help='每批分析数量')
    parser.add_argument('--delay', '-d', type=float, default=1.0,
                        help='请求间隔（秒）')
    parser.add_argument('--summary-only', '-s', action='store_true',
                        help='仅生成汇总报告（不重新分析）')

    args = parser.parse_args()

    analyzer = TemplatePrescriptionAnalyzer()

    if args.summary_only:
        # 仅生成汇总报告
        analyzer.generate_summary_report(args.output)
    else:
        # 完整分析流程
        summary = analyzer.analyze_all_templates(
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            delay=args.delay
        )

        print(f"\n分析完成！")
        print(f"总计模板: {summary['total_templates']}")
        print(f"成功分析: {summary['success_count']}")
        print(f"分析失败: {summary['failed_count']}")

        # 生成汇总报告
        analyzer.generate_summary_report(args.output)


if __name__ == '__main__':
    main()
