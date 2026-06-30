"""
medical/data/shidazhuo/线下诊疗数据采集_20260603151959.xlsx , medical/data/shidazhuo/门诊病人(处方明细)_20260608173446.xlsx 结合这两个表，生成类似medical/data/wuweiping_record_20260525.json中的数据，表格字段说明： 
线下诊疗数据表：
费用明细表：项目代码C开头的是饮片、X开头的是西药、K开头的是颗粒、Z开头是中成药。“剂量”：每个药材的剂量。
首先两个表根据门诊号做关联。
最终提取： 序号、就诊科室、门诊号、姓名、性别、年龄、主诉、现病史、过敏史、既往史、体格检查、辅助检查、中医诊断、西医诊断、处方明细（ps）、频次、治疗计划及建议

1. 中药饮片：根据处方明细，按照频次、用法，生成中药饮片处方。
2. 西药：根据处方明细，按照频次、用法，生成西药处方。
3. 颗粒：根据处方明细，按照频次、用法，生成颗粒处方。
"""
import pandas as pd


def read_precription(file_path:str="medical/data/shidazhuo/门诊病人(处方明细)_20260608173446.xlsx"):
    data = pd.read_excel(file_path, index_col=0,skiprows=2)
    # 
    return data


def read_record(file_path:str="medical/data/shidazhuo/线下诊疗数据采集_20260603151959.xlsx"):
    data = pd.read_excel(file_path, index_col=0)
    print(data)
    import pdb
    pdb.set_trace()
    return data

def main():
    # 读取患者病历
    record_data = read_record()
    # 读取患者处方信息
    prescription_data = read_precription('medical/data_utils/shidazhuo.txt')

    # 根据就诊时间，聚合患者病历和处方信息并存入新的记录 最终存入新的json表中

if __name__ == '__main__':
    main()