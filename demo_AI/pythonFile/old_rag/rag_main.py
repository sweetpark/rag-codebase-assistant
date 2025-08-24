import os
import sys
import pickle
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
    resp = client.embeddings.create(model="text-embedding-3-small", input=query)
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
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return resp.choices[0].message.content

# Extract multiple code blocks: html, controller, service, mapper
def extract_blocks(output):
    # find all fenced blocks
    fences = re.findall(r"```(\w+)\n([\s\S]*?)```", output)
    html = None
    controller = None
    service = None
    mapper = None
    # collect Java blocks
    java_blocks = [code for lang, code in fences if lang == 'java']
    if java_blocks:
        controller = java_blocks[0].strip()
    if len(java_blocks) > 1:
        service = java_blocks[1].strip()
    # HTML block
    for lang, code in fences:
        if lang == 'html':
            html = code.strip()
        if lang == 'xml':
            mapper = code.strip()
    return html, controller, service, mapper

# Save to files
def save_files(html, controller, service, mapper, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if html:
        with open(os.path.join(output_dir, "NewSearch.html"), "w", encoding="utf-8") as f:
            f.write(html)
    if controller:
        with open(os.path.join(output_dir, "TransactionController.java"), "w", encoding="utf-8") as f:
            f.write(controller)
    if service:
        with open(os.path.join(output_dir, "TransactionService.java"), "w", encoding="utf-8") as f:
            f.write(service)
    if mapper:
        with open(os.path.join(output_dir, "TransactionMapper.xml"), "w", encoding="utf-8") as f:
            f.write(mapper)

# Main entry
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    idx_path, metas_path, query = sys.argv[1], sys.argv[2], sys.argv[3]
    top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    out_dir = sys.argv[5] if len(sys.argv) > 5 else "output"

    system_prompt = (
        "당신은 우리 회사의 Java/Spring 전문가입니다.\n"
        "프로젝트 아키텍처:\n"
        "- Frontend: HTML 템플릿 (search.html 스타일)\n"
        "- Controller: MainController.java /account 핸들링\n"
        "- Service: AccountService.execute(oper, payload) 분기\n"
        "- Mapper: AccountMapper.xml MyBatis 쿼리"
    )
    index, metas = load_index_and_metas(idx_path, metas_path)
    retrieved = retrieve(index, metas, query, top_k)
    context = build_context(retrieved)
    user_prompt = f"## Retrieved Context\n{context}\n## Instruction\n{query}"
    output = generate_code(system_prompt, user_prompt)
    html, ctrl, svc, map_sql = extract_blocks(output)
    save_files(html, ctrl, svc, map_sql, out_dir)
    print(f"Generated files saved in '{out_dir}'")
