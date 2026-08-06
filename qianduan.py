import streamlit as st
import requests
Backend_url = "http://localhost:8000"
st.title("RAG应用")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload" not in st.session_state:
    st.session_state.upload=set()
with st.sidebar :
    upload_files = st.file_uploader(
        label="请上传文件",
        type=["pdf","txt","doc","docx"],
        accept_multiple_files=True
    )
    if upload_files:
        for files in upload_files:
            if files.name not in st.session_state.upload:
                st.session_state.upload.add(files.name)
                with st.spinner("正在上传文件"):
                    upload_file = {"files":(files.name,files.getvalue(),files.type)}
                    try:
                        resp = requests.post(f"{Backend_url}/upload",files=upload_file,timeout=1000)
                        if resp.status_code==200:
                            st.success("文件上传成功")
                        else:
                            st.error("后端没有回应")
                    except Exception as e:
                        st.error(f"后端没有回应，返回：{e}")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
question = st.chat_input("请输入问题")
if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("users"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("正在思考"):
            try:
                resp = requests.post(f"{Backend_url}/ask",json={"question":question,"history":st.session_state.messages[:-1]},timeout=1000)
                if resp.status_code==200:
                    answer = resp.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role":"assistant","content":answer})
                else:
                    st.error("请求失败")
            except Exception as e:
                st.error(f"无法连接后端服务：{e}")

