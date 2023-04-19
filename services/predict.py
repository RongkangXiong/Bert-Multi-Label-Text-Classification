# --coding:utf-8--
import hashlib
import random

import torch
from torch.utils.data import TensorDataset, DataLoader

from DataProcess.utils.filehelper import join_prompt_strings
from pybert.io.albert_processor import InputFeature
from pybert.io.bert_processor import InputExample
import time

from pybert.io.utils import collate_fn


def datalabel_to_exampleline(prompt="对比相同和相似物资:", str_a='', str_b='', label=(-1, -1)):
    return [(join_prompt_strings(prompt=prompt, str_a=str_a, str_b=str_b), list(label))]


def sha256_encode(string):
    # 将字符串编码为字节类型
    string_bytes = string.encode('utf-8')
    # 进行SHA-256编码
    sha256_hash = hashlib.sha256(string_bytes)
    # 获取编码结果，并以十六进制形式表示
    result = sha256_hash.hexdigest()
    return result


def text_lines_to_examples(lines: list, example_type: str = 'predict'):
    examples = []
    for i, line in enumerate(lines):
        text_a = line[0]
        label = line[1]
        guid = '%s-%s-%d-%s' % (time.strftime("%Y%m%d%H%M%S", time.localtime()), example_type, i, sha256_encode(text_a)[0:6])
        if isinstance(label, str):
            label = [float(x) for x in label.split(",")]
        else:
            label = [float(x) for x in list(label)]
        text_b = None
        example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        examples.append(example)
    return examples


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_len=512):
    features = []
    for ex_id, example in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        label_id = example.label

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            truncate_seq_pair(tokens_a, tokens_b, max_length=max_seq_len - 3)
        else:
            if len(tokens_a) > max_seq_len - 2:
                tokens_a = tokens_a[:max_seq_len - 2]
        tokens = ['[CLS]'] + tokens_a + ['[SEP]']
        segment_ids = [0] * len(tokens)
        if tokens_b:
            tokens += tokens_b + ['[SEP]']
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_len - len(input_ids))
        input_len = len(input_ids)

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        feature = InputFeature(input_ids=input_ids,
                               input_mask=input_mask,
                               segment_ids=segment_ids,
                               label_id=label_id,
                               input_len=input_len)
        features.append(feature)
    return features


def create_dataset(features, is_sorted=False):
    # Convert to Tensors and build dataset
    if is_sorted:
        print("sorted data by th length of input")
        features = sorted(features, key=lambda x: x.input_len, reverse=True)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_input_lens)
    return dataset


def data_predict(str_a: str = '', str_b: str = '',
                 model=None, tokenizer=None, device=None,
                 max_seq_len=512,
                 example_type='predict',
                 prompt: str = "对比相同和相似物资:"):
    """
    预测函数，输入两个字符串，输出对比结果概率

    Args:
        str_a: 字符串a
        str_b: 字符串b
        model: 训练好的模型
        tokenizer: 分词器
        device: 运行设备，如cpu或gpu
        max_seq_len: 最大序列长度
        example_type: 示例类型
        prompt: 模型输入的前缀

    Returns:
        examples: 示例列表
        logits: 模型输出的对比结果概率

    """
    # 将两个字符串组合为一个示例列表
    example_line = datalabel_to_exampleline(prompt=prompt, str_a=str_a, str_b=str_b)
    # 输出：[('对比相同和相似物资:丽施美首维拼装式刮砂防滑地垫转角4*5cm黑色单位：块,Raxwell疏水防滑垫S型镂空加密PVC1.2m*1m*5mm灰色单位：片', [-1, -1])]
    examples = text_lines_to_examples(example_line, example_type=example_type)

    # 将示例列表转化为特征列表
    test_features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, max_seq_len=max_seq_len)
    # 将特征列表转化为数据集
    test_dataset = create_dataset(test_features)
    # 将数据集转化为数据加载器
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    for step, batch in enumerate(test_dataloader):
        # 将批次数据转移到对应设备上
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        # 将模型设置为评估模式
        model.eval()
        with torch.no_grad():
            # 计算模型的输出结果
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()
            return examples, logits
