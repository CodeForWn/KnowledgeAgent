# 使用官方Python镜像作为基础镜像
FROM python:3.10-slim

# 设置容器内的工作目录
WORKDIR /app

# 将requirements.txt拷贝到容器内并安装依赖
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 拷贝app目录到容器内
COPY app ./app

# 暴露服务端口（根据你的实际项目调整，这里假设8000）
EXPOSE 8000

# 启动命令（与你当前的启动命令保持一致）
CMD ["python", "app/KMC_app.py"]


