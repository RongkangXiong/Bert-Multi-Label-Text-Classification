# --coding:utf-8--
from pathlib import Path

BASE_DIR = Path('services')
print(BASE_DIR)

config = {
    'model_path': "./pybert/output/checkpoints/bert",
    # 'model_path': 'T:\WorkSpace\\NLP\Bert-Multi-Label-Text-Classification\pybert\output\checkpoints\\bert',
    'vocab_path': './pybert/pretrain/bert/bert-uncased/vocab.txt',
    'num_labels': 2,
    'do_lower_case': True,
    'max_seq_len': 512,
    'log_path': "./services/logs",
    'index': "./services/index.html",
    "api_key": "123456"
}
