import os
os.environ["HF_HUB_OFFLINE"] = "1"
from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader,UnstructuredWordDocumentLoader
from mineru_parser import parse_pdf_with_mineru
import asyncio,tempfile,traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from operator import itemgetter
from langchain_core.messages import HumanMessage,AIMessage
from rank_bm25 import BM25Okapi 
from sentence_transformers import CrossEncoder
import jieba
import pickle
from pymilvus import MilvusClient,DataType
app = FastAPI()
Collection_Name = "rag_collection"
embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh:latest")
client = MilvusClient(uri="http://127.0.0.1:19530")
All_chunks = []
BM25_index = None
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3",device="cuda")
def get_or_create_collection():
    if client.has_collection(Collection_Name):
        return 
    schema = client.create_schema(auto_id = True,enable_dynamic_field=False)
    schema.add_field(field_name="id",datatype=DataType.INT64,is_primary=True)
    schema.add_field(field_name="text",datatype=DataType.VARCHAR,max_length=5000)
    schema.add_field(field_name="embedding",datatype=DataType.FLOAT_VECTOR,dim=768)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="embedding",index_type="AUTOINDEX",metric_type = "COSINE")
    client.create_collection(collection_name=Collection_Name,schema=schema,index_params=index_params)
class AnswerResponse(BaseModel):
    question:str
    answer:str
class AskResponse(BaseModel):
    question:str
    history:list = []
@app.post("/upload")
async def upload_file(
    files :UploadFile=File(...)
):
    temp_path = None
    global All_chunks,BM25_index
    try:
        suffix = os.path.splitext(files.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False,
        suffix=suffix) as temp_file:
            content = await files.read()
            temp_file.write(content)
            temp_path = temp_file.name
        if files.content_type=="application/pdf":
            # MinerU 解析复杂PDF（表格结构完整、质量高），丢线程池避免阻塞事件循环
            docs = await asyncio.to_thread(parse_pdf_with_mineru, temp_path)
        elif files.content_type=="text/plain":
            # 逐个尝试编码，找到能成功加载的就用
            docs = None
            for enc in ["utf-8", "gbk", "gb18030", "utf-16"]:
                try:
                    loader = TextLoader(temp_path, encoding=enc)
                    docs = loader.load()
                    break
                except:
                    continue
            if docs is None:
                raise HTTPException(status_code=400, detail="无法识别 txt 文件编码，请另存为 UTF-8 后重试")
        else:
            loader = UnstructuredWordDocumentLoader(temp_path)
        if 'docs' not in dir():
            docs = loader.load()
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 80,
            separators=["\n\n","\n","。","？","，"," "]
        )
        chunks = text_spliter.split_documents(docs)
        vectors = embeddings.embed_documents([d.page_content for d in chunks])
        get_or_create_collection()
        client.insert(collection_name=Collection_Name,data=[{"text":d.page_content,"embedding":vectors[i]}for i,d in enumerate(chunks)])
        All_chunks.extend(chunks)
        tokenized_docs = [list(jieba.cut(d.page_content)) for d in All_chunks]
        BM25_index = BM25Okapi(tokenized_docs,k1=1.5,b=0.75)
        with open("chunks.pkl","wb") as f:
            pickle.dump(All_chunks,f)
        return {"messages":"文件上传成功","filesname":files.filename}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
@app.post("/ask")
async def ask_question(
    rep:AskResponse
):
    global All_chunks,BM25_index,reranker
    if not All_chunks:
        raise HTTPException(status_code=500,detail="请上传文件")
    try:
        tokenized_q = list(jieba.cut(rep.question))
        bm25_scores = BM25_index.get_scores(tokenized_q)
        top_bm25 = sorted(
            range(len(bm25_scores)),
            key=lambda i:bm25_scores[i],
            reverse=True
        )[:10]
        bm25_docs = [All_chunks[i] for i in top_bm25 if bm25_scores[i]>0]
        query_vec = embeddings.embed_query(rep.question)
        get_or_create_collection()
        res = client.search(
                    collection_name=Collection_Name,
                    data=[query_vec],
                    limit=10,
                    output_fields=["text"],
                    search_params={"metric_type":"COSINE"}
                )
        milvus_texts = [h["entity"]["text"]for h in res[0]]
        milvus_docs =[]
        for t in milvus_texts:
            for d in All_chunks:
                if d.page_content == t and d not in milvus_docs:
                    milvus_docs.append(d)
                    break
        seen =set()
        final_docs =[]
        for doc in bm25_docs+milvus_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                final_docs.append(doc)
        pairs = [[rep.question,doc.page_content]for doc in final_docs]
        scores = reranker.predict(pairs)
        scored = list(zip(final_docs,scores))
        scored.sort(key=lambda x: x[1],reverse=True)
        top_reranked = scored[:5]
        final_docs = [doc for doc,score in top_reranked]
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system","必须根据上下文回答，没有根据就说不知道，上下文内容为：{content}"),
                MessagesPlaceholder(variable_name="history"),
                ("human","{question}")
            ]
        )
        llm = ChatOllama(model="qwen3:1.7b")
        rag_chain=(
            {
                "content": lambda _:format_docs(final_docs),
                "question": itemgetter("question"),
                "history": itemgetter("history")
            }
            |prompt
            |llm
            |StrOutputParser()
        )
        chat_history = []
        for msg in rep.history:
            if msg["role"]=="user":
                chat_history.append(HumanMessage(msg["content"]))
            elif msg["role"]=="assistant":
                chat_history.append(AIMessage(msg["content"]))
        answer = rag_chain.invoke({"question":rep.question,"history":chat_history})
        return AnswerResponse(question=rep.question,answer=answer)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500,detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8600)
