# -*- coding: utf-8 -*-
"""
对比三种解析方式对同一份复杂PDF（含表格+图片）的处理效果：
  A. PyPDFLoader  —— 你现在的方案，纯文本抽取
  B. Unstructured —— 元素流（fast 免模型 / hi_res 需模型+OCR）
  C. MinerU       —— Markdown 方案（已用 CLI 跑完，见 mineru_out/）

运行: python compare_parsers.py
"""
import os
# 国内网络：huggingface.co 不可直连。hi_res 的 yolox 模型已手动放入本地缓存
# （见会话记录），离线加载。若第一次跑报模型找不到，删掉 HF_HUB_OFFLINE 改走
# HF_ENDPOINT=https://hf-mirror.com，但新版 huggingface_hub 会拦截镜像重定向。
os.environ["HF_HUB_OFFLINE"] = "1"

PDF = "sample_complex.pdf"


def demo_pypdf():
    print("=" * 60)
    print("A. PyPDFLoader（纯文本抽取）")
    print("=" * 60)
    from langchain_community.document_loaders import PyPDFLoader
    docs = PyPDFLoader(PDF).load()
    print(f">>> {len(docs)} 个 Document，正文如下（注意表格/图片）：\n")
    print(docs[0].page_content[:400])


def demo_unstructured(strategy):
    print("=" * 60)
    print(f"B. Unstructured（strategy={strategy}）")
    print("=" * 60)
    from unstructured.partition.pdf import partition_pdf
    try:
        kwargs = dict(filename=PDF, strategy=strategy)
        if strategy == "hi_res":
            kwargs["extract_images_in_pdf"] = True
            kwargs["extract_image_block_output_dir"] = "out_unstructured_images"
        elements = partition_pdf(**kwargs)
    except Exception as e:
        print(f">>> hi_res 失败: {type(e).__name__}: {e}")
        print(">>> 说明: 此版本 hi_res 强依赖 Tesseract OCR，Windows 需安装后设置 PATH。")
        print(">>> 已降级用 fast 策略演示（表格识别不了，见下方）。\n")
        elements = partition_pdf(filename=PDF, strategy="fast")

    print(f">>> {len(elements)} 个元素（属性名是 category，不是 type）:\n")
    for el in elements:
        if el.category == "Table":
            print(f"[{el.category}] 结构HTML: {el.metadata.text_as_html}")
        else:
            print(f"[{el.category}] {el.text[:40]!r}")
    if strategy == "hi_res":
        print(">>> 图片已保存到 out_unstructured_images/")
        for f in os.listdir("out_unstructured_images"):
            print("    -", f)


def show_mineru():
    print("=" * 60)
    print("C. MinerU（CLI 输出，mineru_out/）")
    print("=" * 60)
    import glob
    md = glob.glob("mineru_out/**/*.md", recursive=True)
    if not md:
        print(">>> 尚未运行 mineru -p sample_complex.pdf -o mineru_out -b pipeline -m auto")
        return
    print(">>> 产物文件：", "、".join(glob.glob("mineru_out/**/*", recursive=True)[:6]), "...")
    print(">>> Markdown 内容（表格是完整 HTML <table>，图片用相对路径引用）：\n")
    with open(md[0], encoding="utf-8") as f:
        print(f.read()[:800])


if __name__ == "__main__":
    demo_pypdf()
    print()
    demo_unstructured("hi_res")
    print()
    show_mineru()
