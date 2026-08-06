FROM python:3.11-slim

WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --default-timeout=120

# 复制项目代码
COPY ragproject.py .
COPY qianduan.py .

# 暴露后端和前端端口
EXPOSE 8000 8501

# 启动后端和前端
CMD ["sh", "-c", "uvicorn ragproject:app --host 0.0.0.0 --port 8000 & streamlit run qianduan.py --server.port 8501 --server.address 0.0.0.0"]
