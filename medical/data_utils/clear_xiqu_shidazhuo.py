"""

"""
import pandas as pd


def read_precription(file_path:str="medical/data/shidazhuo/门诊病人费用明细(处方)_20260603100719.xlsx"):
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