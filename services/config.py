# --coding:utf-8--
from pathlib import Path
import torch
BASE_DIR = Path('services')
print(BASE_DIR)




config = {
    "device": "cuda:1",
    'model_path': "./pybert/output/checkpoints/bert",
    # 'model_path': 'T:\WorkSpace\\NLP\Bert-Multi-Label-Text-Classification\pybert\output\checkpoints\\bert',
    'vocab_path': './pybert/pretrain/bert/bert-uncased/vocab.txt',
    'num_labels': 2,
    'do_lower_case': True,
    'max_seq_len': 512,
    'log_path': "./services/logs",
    'index': "./services/index.html",
    "api_key": "123456",
}


def get_device():
    try:
        device = config['device']
    except Exception as e:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device