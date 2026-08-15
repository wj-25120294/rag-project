# -*- coding: utf-8 -*-
"""
MinerU 解析 PDF 的封装。写一次，其他地方直接调用一行即可。

用法:
    from mineru_parser import parse_pdf_with_mineru
    docs = await asyncio.to_thread(parse_pdf_with_mineru, pdf_path)
    # docs 是 list[langchain Document]，page_content 是含完整 <table> 的 markdown
"""
import os
import re
import shutil
import subprocess
import tempfile

from langchain_core.documents import Document


def parse_pdf_with_mineru(pdf_path: str, timeout: int = 1800) -> list:
    """用 MinerU 把 PDF 解析成 markdown，返回单个 langchain Document。

    - 表格：保留完整 HTML <table> 结构（MinerU 质量最高）
    - 图片：MinerU 会抽出到 images/，但纯文本 RAG 用不上，这里去掉图片引用
    - timeout: 秒，MinerU 处理大文件较慢，默认 30 分钟
    """
    out_dir = tempfile.mkdtemp(prefix="mineru_")
    try:
        subprocess.run(
            ["mineru", "-p", pdf_path, "-o", out_dir,
             "-b", "pipeline", "-m", "auto"],
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        md_path = os.path.join(out_dir, stem, "auto", f"{stem}.md")
        if not os.path.exists(md_path):
            raise RuntimeError(f"MinerU 未产出预期文件: {md_path}")
        with open(md_path, encoding="utf-8") as f:
            md = f.read()
        # 去掉图片引用 ![...](...)，纯文本检索用不上，避免浪费 token
        md = re.sub(r"!\[.*?\]\(.*?\)", "", md)
        return [Document(page_content=md)]
    except subprocess.TimeoutExpired:
        raise RuntimeError("MinerU 解析超时，请检查 PDF 大小或增大 timeout 参数")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MinerU 解析失败: {e.stderr.decode('utf-8', 'ignore')[-500:]}") from e
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == "__main__":
    # 单独测试: python mineru_parser.py sample_complex.pdf
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_complex.pdf"
    docs = parse_pdf_with_mineru(pdf)
    print(f"解析完成, {len(docs)} 个 Document, 共 {len(docs[0].page_content)} 字符")
    print("--- 前 300 字符 ---")
    print(docs[0].page_content[:300])
