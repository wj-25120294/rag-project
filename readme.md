# RAG 智能问答系统（第二版）

基于 **FastAPI + Milvus + LangChain + Ollama** 的中文 RAG 问答系统。第二版将 PDF 解析升级为 **MinerU**，能完整保留复杂 PDF 中的**表格结构**、提取图片，显著提升表格类问题的检索准确率。

## 系统流程

```
上传PDF ──> MinerU解析(markdown+完整<table>) ──> 800字切分 ──> 向量化(Ollama)
                                                    │
   提问 ──> 混合检索(BM25 关键词 + 向量相似度) ──> 融合去重 ──> reranker精排 ──> LLM生成回答
```

## 技术栈

| 模块 | 技术 |
|---|---|
| 后端 | FastAPI + uvicorn |
| 向量库 | Milvus（docker-compose 部署） |
| 检索 | BM25（jieba 分词）+ Dense Vector 混合，CrossEncoder reranker（BAAI/bge-reranker-v2-m3） |
| 嵌入 / LLM | Ollama（dmeta-embedding-zh / qwen3:1.7b） |
| **PDF 解析（v2 新增）** | **MinerU**（`-b pipeline -m auto`），封装于 `mineru_parser.py` |
| 前端 | Streamlit |

## 第二版更新

1. **PDF 解析升级**：`PyPDFLoader`（纯文本，表格拆散、图片丢失）→ **MinerU**（表格结构完整、质量高）。
2. **修复历史 bug**：`page_contetn` → `page_content`（Milvus 插入）、`metric_type="CONSINE"` → `"COSINE"`（Milvus 搜索）。
3. **端口调整**：后端 `8000`→`8600`、Streamlit `8501`→`8602`（Windows 保留端口段问题，见 `.streamlit/config.toml` 与注释）。
4. **新增脚本**：`mineru_parser.py`（MinerU 封装）、`compare_parsers.py`（PyPDFLoader / Unstructured / MinerU 三种解析方案对比学习）。

## 运行方法

```bash
# 1. 启动 Milvus
docker compose -f milvus/docker-compose.yml up -d

# 2. 启动 Ollama 并确认模型已拉取
ollama pull shaw/dmeta-embedding-zh:latest
ollama pull qwen3:1.7b

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动后端（端口 8600）
python ragproject.py

# 5. 启动前端（端口 8602）
streamlit run qianduan.py
```

> 注意：模型（reranker、MinerU、BAAI/bge-reranker-v2-m3 等）需要网络下载；国内网络直连 huggingface.co 可能失败，请配置 `HF_ENDPOINT=https://hf-mirror.com` 或使用本地离线缓存（本项目脚本已设 `HF_HUB_OFFLINE=1`）。

## 目录结构

```
├── ragproject.py          # FastAPI 后端（上传 / 问答）
├── qianduan.py            # Streamlit 前端
├── mineru_parser.py       # MinerU PDF 解析封装（v2）
├── test.py                # RAG 评估（ragas：faithfulness / relevancy / context）
├── feishu_bot.py          # 飞书机器人（DeepSeek 生成 PPT）
├── compare_parsers.py     # PyPDFLoader / Unstructured / MinerU 对比
├── milvus/docker-compose.yml   # Milvus 部署
├── requirements.txt
└── .streamlit/config.toml # Streamlit 端口配置
```

## 学习记录

由浅入深的学习路线：

1. **FastAPI**：GET/POST、路径与查询参数、请求体、Pydantic 校验。
2. **LangChain RAG 链**：加载 → 切分 → 向量化 → 索引 → 问答。
3. **查询变换**：HyDE、Multi-Query。
4. **混合检索**：BM25（jieba）+ Dense Vector + CrossEncoder reranker。
5. **RAG 评估**：ragas（Faithfulness、Answer Relevancy、Context Precision、Context Recall）。
6. **高级 PDF 解析**（v2）：PyPDFLoader vs Unstructured vs MinerU，表格/图片的处理差异。
