FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive
ENV TERM linux

RUN apt-get update && apt-get install -y python3.8
RUN pip install fonttools -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get update && apt-get install -y apt-utils

# 安装依赖包
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    vim \
    apt-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

# # 设置时区
RUN ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo Asia/Shanghai > /etc/timezone

# 升级 pip 并安装依赖包
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
COPY requirements_server.txt /app/
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install -r /app/requirements_server.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码到镜像
COPY . /app/

# 设置工作目录和环境变量
WORKDIR /app
ENV PYTHONPATH=/app

# 暴露端口并启动命令
EXPOSE 8000
CMD ["uvicorn", "services.app:app", "--host", "0.0.0.0", "--port", "8000"]
