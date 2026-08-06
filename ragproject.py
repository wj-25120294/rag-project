from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader,TextLoader,UnstructuredWordDocumentLoader
import tempfile,os,traceback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings,ChatOllama
from operator import itemgetter
from langchain_core.messages import HumanMessage,AIMessage
from rank_bm25 import BM25Okapi 
from sentence_transformers import CrossEncoder
import jieba
import pickle
app = FastAPI()
Vector =None
All_chunks = []
BM25_index = None
os.environ["HF_HUB_OFFLINE"] = "1"
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3",device="cuda")
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
    global Vector,All_chunks,BM25_index
    try:
        suffix = os.path.splitext(files.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False,
        suffix=suffix) as temp_file:
            content = await files.read()
            temp_file.write(content)
            temp_path = temp_file.name
        if files.content_type=="application/pdf":
            loader = PyPDFLoader(temp_path)
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
        try:
            embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh:latest")
        except:
            raise HTTPException(
                status_code=400,
                detail="Ollama服务未启用"
            )
        
        if Vector is None:
            Vector = FAISS.from_documents(documents=chunks, embedding=embeddings)
            All_chunks = chunks
        else:
            Vector.add_documents(documents=chunks)
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
    global Vector,All_chunks,BM25_index,reranker
    if Vector is None:
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
        faiss_docs = []
        for doc,score in Vector.similarity_search_with_score(rep.question,k=10):
            faiss_docs.append(doc)
        seen =set()
        final_docs =[]
        for doc in bm25_docs+faiss_docs:
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
