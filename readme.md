# 这是我开始学习大模型应用开发的代码



## 1.先学了fastapi的使用：get,post请求，路径参数，查阅参数，请求体，pydantic字段检验（Field,Path,Query）




## 2.学了langchain中rag链的构建：加载，分割，向量化，创建向量索引，naive rag系统




## 3.学了Query Transformation,主要是Hyde（先让llm写答案，再让其拿答案搜索），multi-query（让llm生成多个问题，搜索答案），两者可以结合，先multi-query,再Hyde




## 4.学了bm25+embedding混合检索（先切分文档，根据字频，关键字打分，每个文档与问题计算相关性，并排序），还要reranker（把问题和文档结合，打分但是reranker会结合语义，所以更精准也更慢），详细见代码




## 5.学了如何评估rag体系，并将原先单一相似性检索和混合检索做了对比，由于文档的简单性，两者content_precision,content_recall基本相同，单一相似性检索的faithfulness有0.9500，answer_relevancy有0.9518，混合排序加上reranker的faithfulness有0.9333，answer_relevancy有0.9452，