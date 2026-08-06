import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 国内镜像

from huggingface_hub import snapshot_download

snapshot_download("BAAI/bge-reranker-v2-m3",
                    local_dir=r"D:\xunlei\rag项目\models\bge-reranker")
print("完成")