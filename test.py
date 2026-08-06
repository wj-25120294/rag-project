import os
os.environ["HF_HUB_OFFLINE"] = "1"
import pickle
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama,OllamaEmbeddings
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import EvaluationDataset,SingleTurnSample,evaluate
from ragas.metrics import Faithfulness,ResponseRelevancy,ContextPrecision,ContextRecall
from langchain_deepseek import ChatDeepSeek
with open ("chunks.pkl","rb") as f:
    All_chunks = pickle.load(f)
embeddings = OllamaEmbeddings(model="shaw/dmeta-embedding-zh:latest",base_url="http://127.0.0.1:11434")
Vector = FAISS.from_documents(documents=All_chunks,embedding=embeddings)
tokenized_docs = [list(jieba.cut(d.page_content)) for d in All_chunks]
bm25_index = BM25Okapi(tokenized_docs,k1=1.5,b=0.75)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3",device="cuda")

def retriever(question):
    # tokenized_q = list(jieba.cut(question))
    # bm25_score = bm25_index.get_scores(tokenized_q)
    # top_bm25 = sorted(
    #     range(len(bm25_score)),
    #     key= lambda i:bm25_score[i],
    #     reverse=True
    # )[:10]
    # bm25_docs = [All_chunks[i] for i in top_bm25 if bm25_score[i]>0]
    # faiss_docs = [doc for doc,scores in Vector.similarity_search_with_score(question,k=10)]
    # seen = set()
    # final_docs = []
    # for doc in bm25_docs+faiss_docs:
    #     if doc.page_content not in seen:
    #         seen.add(doc.page_content)
    #         final_docs.append(doc)
    # pairs = [[question,doc.page_content]for doc in final_docs]
    # scores = reranker.predict(pairs)
    # scored = sorted(zip(final_docs,scores),key=lambda x: x[1],reverse=True)
    # return [doc for doc,_ in scored[:5]]
    retrievers  = Vector.as_retriever(search_type = "similarity",search_kwargs = {"k":5})
    return retrievers.invoke(question)
def generate(question,retrieved_docs,history = None):
    prompt = ChatPromptTemplate.from_messages([
        ("system","必须根据上下文回答，没有根据就说不知道，上下文内容为：{content}"),
        ("human","{question}")
    ])
    llm = ChatOllama(model="qwen3:1.7b")
    chain = prompt|llm|StrOutputParser()
    content = "\n\n".join(d.page_content for d in retrieved_docs)
    return chain.invoke({"question":question,"content":content})

# 手写测试问题 + 参考答案（参考答案来自 chunks 原文，供 ContextRecall 使用）
# 每项格式: (问题, 参考答案)
qa_pairs = [
    ("软科2026年主榜中排名第一的大学是哪所？总分是多少？", "清华大学，总分1087.1分。"),
    ("北京大学在软科2026年主榜中排名第几？", "北京大学排名第二，总分1036.3分。"),
    ("构成“上海双雄”的是哪两所大学？", "上海交通大学与复旦大学。"),
    ("中国科学技术大学位于哪个城市？隶属于什么机构？", "位于安徽合肥，隶属于中国科学院。"),
    ("QS世界大学排名中稳居全球前20名的中国大学是哪两所？", "北京大学和清华大学。"),
    ("2025年GDP超过5万亿元的城市有哪些？", "上海和北京，是仅有的两个GDP超过5万亿元的城市。"),
    ("2026年度中国百强城市指数的一线城市是哪四个？", "北京、上海、广州、深圳。"),
    ("2025年GDP突破万亿元的城市是哪两个？", "温州和大连，突破后中国万亿GDP城市总数达到29个。"),
    ("2025年GDP前二十强中增速第一的是哪个城市？增速是多少？", "合肥，增速为6.1%，排名第一。"),
    ("人均GDP排名反映的是什么指标？哪些大城市的人均GDP位居前列？", "人均GDP反映产出效率；主要大城市中，北京和上海的人均GDP位居前列。"),
]

samples = []
for q, reference in qa_pairs:
    docs = retriever(q)
    contexts = [d.page_content for d in docs]
    answer = generate(q,docs)
    samples.append(SingleTurnSample(
        user_input=q,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    ))
dataset = EvaluationDataset(samples=samples)
result = evaluate(
    dataset,
    metrics=[Faithfulness(),ResponseRelevancy(),ContextPrecision(),ContextRecall()],
    llm = ChatDeepSeek(model="deepseek-chat",temperature=0.2,api_key="sk-02ee692ab01e46e68a51c19138055ff3"),
    embeddings=embeddings
)
print(result)



