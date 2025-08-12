
"""
Usage:
  # 1) 벡터 인덱스 만들기
  python VectorStoreAndRetrieval.py build_index \
      output/embeddings.jsonl \
      output/code_index.faiss \
      output/metas.pkl

  # 2) 질의 실행하기
  python VectorStoreAndRetrieval.py query \
      output/code_index.faiss \
      output/metas.pkl \
      "거래 데이터 조회 화면 생성" \
      5
"""

import json
import pickle
import sys
import numpy as np
import faiss
import openai

def build_index(embeddings_path, index_path, metas_path):
    # 1) embeddings.jsonl 읽기
    metas = []
    vectors = []
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            metas.append({'id': obj['id'], 'role': obj['role'], 'text': obj['text']})
            vectors.append(obj['embedding'])
    vectors = np.array(vectors, dtype='float32')
    dim = vectors.shape[1]

    # 2) FAISS 인덱스 생성 (L2 거리)
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # 3) 결과 저장
    faiss.write_index(index, index_path)
    with open(metas_path, 'wb') as m:
        pickle.dump(metas, m)

    print(f"Built index with {len(vectors)} vectors (dim={dim})")

def query_index(index_path, metas_path, query, top_k):
    # 1) 인덱스·메타로드
    index = faiss.read_index(index_path)
    metas = pickle.load(open(metas_path, 'rb'))

    # 2) 질의 임베딩
    resp = openai.embeddings.create(model="text-embedding-ada-002", input=query)
    qvec = np.array(resp.data[0].embedding, dtype='float32').reshape(1, -1)

    # 3) 검색
    distances, indices = index.search(qvec, top_k)
    results = [metas[i] for i in indices[0]]
    return results

def main():
    cmd = sys.argv[1]
    if cmd == 'build_index':
        _, _, emb_path, idx_path, metas_path = sys.argv
        build_index(emb_path, idx_path, metas_path)
    elif cmd == 'query':
        _, _, idx_path, metas_path, query, k = sys.argv
        for r in query_index(idx_path, metas_path, query, int(k)):
            print(f"[{r['role']}] {r['id']}\n{r['text']}\n")
    else:
        print("Usage:\n  build_index <emb.jsonl> <index.faiss> <metas.pkl>\n  query <index.faiss> <metas.pkl> <query> <top_k>")

if __name__ == '__main__':
    main()