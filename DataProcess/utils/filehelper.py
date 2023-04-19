# --coding:utf-8--
import csv

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_csv_file(file_path, encoding='utf-8-sig'):
    data = []
    with open(file_path, 'r', encoding=encoding) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in tqdm(csv_reader):
            data.append(row)
    return data


def clean_string(s):
    """
    去除字符串中的空格和标点符号

    Args:
        s (str): 待处理的字符串

    Returns:
        str: 处理后的字符串
    """
    s = s.replace(" ", "").replace("'", "").replace('"', '').replace(",", "").replace("，", "")  # 去除所有空格中英文逗号和引号
    return s


def join_prompt_strings(prompt: str = "对比相同和相似物资:", str_a: str = None, str_b: str = None):
    return prompt + ",".join([clean_string(str_a), clean_string(str_b)])
