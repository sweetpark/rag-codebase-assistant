"""
rag_main.py

Step 5 & 6: Retrieval + Generation + Post-processing
FAISS index (.faiss) and metadata (.pkl) must exist.

Usage:
  python rag_main.py <index.faiss> <metas.pkl> "<query>" [<top_k>] [<output_dir>]
"""
import os
import sys
import pickle
import json
import re
import faiss
import numpy as np
from openai import OpenAI

# Load FAISS index and metadata
def load_index_and_metas(index_path, metas_path):
    index = faiss.read_index(index_path)
    metas = pickle.load(open(metas_path, 'rb'))
    return index, metas

# Retrieve top_k chunks for query
def retrieve(index, metas, query, top_k=5):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.embeddings.create(model="text-embedding-ada-002", input=query)
    qvec = np.array(resp.data[0].embedding, dtype='float32').reshape(1, -1)
    _, I = index.search(qvec, top_k)
    return [metas[i] for i in I[0]]

# Build prompt context grouped by role
def build_context(chunks):
    sections = {}
    for c in chunks:
        sections.setdefault(c['role'], []).append(c['text'])
    context = ""
    for role, texts in sections.items():
        context += f"### {role}\n"
        for t in texts:
            context += t.strip() + "\n\n"
    return context

# Generate code via ChatCompletion
def generate_code(system_prompt, user_prompt):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    return resp.choices[0].message.content

# Extract code blocks from output
def extract_blocks(output):
    html = re.search(r"```html(.*?)```", output, re.DOTALL)
    java = re.search(r"```java(.*?)```", output, re.DOTALL)
    xml  = re.search(r"```xml(.*?)```",  output, re.DOTALL)
    return (
        html.group(1).strip() if html else "",
        java.group(1).strip() if java else "",
        xml.group(1).strip()  if xml  else ""
    )

# Save to files
def save_files(html, java_code, xml_code, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "NewSearch.html"), "w", encoding="utf-8") as f:
        f.write(html)
    with open(os.path.join(output_dir, "TransactionController.java"), "w", encoding="utf-8") as f:
        f.write(java_code)
    with open(os.path.join(output_dir, "TransactionMapper.xml"), "w", encoding="utf-8") as f:
        f.write(xml_code)

# Main entry
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    idx_path   = sys.argv[1]
    metas_path = sys.argv[2]
    query      = sys.argv[3]
    top_k      = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    out_dir    = sys.argv[5] if len(sys.argv) > 5 else "output"

    # System prompt describing our architecture
    system_prompt = (
        "당신은 우리 회사의 Java/Spring 전문가입니다.\n"
        "프로젝트 아키텍처:\n"
        "- Frontend: HTML 템플릿 (search.html 스타일, ag-Grid, data-service/oper 패턴)\n"
        "- Controller: MainController.java 의 /account 처리 방식\n"
        "- Service: AccountService.execute(oper, payload) 분기 로직\n"
        "- Mapper: AccountMapper.xml의 MyBatis SQL 매퍼"
    )

    index, metas = load_index_and_metas(idx_path, metas_path)
    retrieved    = retrieve(index, metas, query, top_k)
    print("=== Retrieved Chunks ===")
    for r in retrieved:
        print(f"[{r['role']}] {r['id']}\n{r['text'][:200]}...\n")
    
    context      = build_context(retrieved)

    user_prompt = f"## Retrieved Context\n{context}\n## Instruction\n{query}"
    output      = generate_code(system_prompt, user_prompt)

    html_blk, java_blk, xml_blk = extract_blocks(output)
    save_files(html_blk, java_blk, xml_blk, out_dir)
    print(f"Generated files saved in '{out_dir}'")
