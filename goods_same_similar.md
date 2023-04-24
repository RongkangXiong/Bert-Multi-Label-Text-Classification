# 下载预训练模型
进入 `cd ./pybert/pretrain/bert`,然后执行下载预训练模型

```shell
# Ubuntu / Debian
$ sudo apt-get install git-lfs
# Fedora / Red Hat
$ sudo dnf install git-lfs
# macOS
$ brew install git-lfs
```


```shell
git lts install
git clone https://huggingface.co/bert-base-multilingual-uncased
```

改名


# preprocess data.
```shell
python run_bert.py --do_data --do_lower_case 
--data_name goods_same_similar 
--train_batch_size=32 
--eval_batch_size=128 
--train_max_seq_len=512 
--eval_max_seq_len=512
```
一行脚本：
```shell
python run_bert.py --do_data --do_lower_case --data_name goods_same_similar --train_batch_size=32 --eval_batch_size=128 --train_max_seq_len=512 --eval_max_seq_len=512
```

# training 

```shell
python run_bert.py --do_train --save_best 
--do_lower_case 
--data_name goods_same_similar 
--epochs=6
--train_batch_size=32 
--eval_batch_size=128 
--train_max_seq_len=512 
--eval_max_seq_len=512
```

一行脚本：
```shell
python run_bert.py --do_train --save_best --do_lower_case --data_name goods_same_similar --epochs=6 --train_batch_size=32 --eval_batch_size=128 --train_max_seq_len=512 --eval_max_seq_len=512
```

# test

```shell
python run_bert.py --do_test 
--do_lower_case 
--data_name goods_same_similar 
--epochs 6 --train_batch_size 512 
--train_max_seq_len 512 
--eval_max_seq_len 512
```

一行脚本:
```shell
python run_bert.py --do_test --do_lower_case --data_name goods_same_similar --epochs 6 --train_batch_size 512 --train_max_seq_len 512 --eval_max_seq_len 512
```

# service 启动

```shell
uvicorn services.app:app --reload --host 0.0.0.0 --workers 6
```

# 导出requirements
```shell
pip freeze > requirements_raw.txt
```

```shell
python filter_dependencies.py requirements_raw.txt requirements.txt
```

# docker build
```shell
nvidia-docker build -t goods_same_similar:v1.0 .
```

```shell
sudo docker-compose up
```