"""
RAG pipeline (OpenAI-only, single-command)
- Looks for ./output/new_chunk.jsonl (or new_chunk.json)
- If FAISS/metas not found, it builds them under ./output
- Then retrieves with OpenAI embeddings and generates HTML/Java/XML files into ./output

Run
---
python new_rag_main.py --query "계정 정보 조회 화면 및 API 생성"

Optional flags
--------------
--outdir ./output                 # default
--embed-model text-embedding-3-small
--chat-model gpt-4o-mini
--topk 6

Requires
--------
- OPENAI_API_KEY in env
- openai>=1.x, faiss-cpu, numpy
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------- deps ----------------

def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError("faiss (faiss-cpu) is required. pip install faiss-cpu") from e


def _openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai>=1.x is required. pip install openai") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI()


# -------------- io helpers --------------

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _resolve_paths(outdir: Path) -> Dict[str, Path]:
    paths = {
        "outdir": outdir,
        "chunks_jsonl": outdir / "rechunk_noisy.jsonl",
        "chunks_json": outdir / "rechunk_noisy.json",
        "embedding": outdir / "embedding.jsonl",
        "index": outdir / "code_index.faiss",
        "metas": outdir / "metas.pkl",
    }
    return paths


def _load_jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8").strip()
    docs: List[Dict[str, Any]] = []
    if text.startswith("["):
        arr = json.loads(text)
        for obj in arr:
            docs.append(obj)
        return docs
    # jsonl
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            docs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return docs


def load_chunks_auto(outdir: Path) -> List[Dict[str, Any]]:
    p = _resolve_paths(outdir)
    if p["chunks_jsonl"].exists():
        return _load_jsonl_or_json(p["chunks_jsonl"])
    if p["chunks_json"].exists():
        return _load_jsonl_or_json(p["chunks_json"])
    raise FileNotFoundError("rechunk.jsonl/json not found in ./output")


def normalize_chunk_keys(obj: Dict[str, Any], i: int) -> Dict[str, Any]:
    # Normalize keys to: id, file_path, lang, type, content, docstring, commit_message, tags
    out = dict(obj)
    out.setdefault("id", obj.get("chunk_id") or obj.get("name") or f"chunk_{i}")
    out.setdefault("file_path", obj.get("path", obj.get("file_path", "")))
    out.setdefault("lang", obj.get("language", obj.get("lang", "")))
    out.setdefault("type", obj.get("kind", obj.get("type", "")))
    out.setdefault("content", obj.get("text", obj.get("content", "")))
    out.setdefault("docstring", obj.get("doc", obj.get("docstring", "")))
    out.setdefault("commit_message", obj.get("commit", obj.get("commit_message", "")))
    out.setdefault("tags", obj.get("labels", obj.get("tags", [])))
    return out


def build_text_for_embedding(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    if item.get("id"): parts.append(f"[ID] {item['id']}")
    if item.get("file_path"): parts.append(f"[FILE] {item['file_path']}")
    t = item.get("type"); l = item.get("lang")
    if t or l: parts.append(f"[META] type={t}, lang={l}")
    tags = item.get("tags") or []
    if tags: parts.append("[TAGS] " + ",".join(map(str, tags)))
    doc = (item.get("docstring") or "").strip()
    if doc: parts.append("[DOC] " + doc)
    commit = (item.get("commit_message") or "").strip()
    if commit: parts.append("[COMMIT] " + commit)
    code = (item.get("content") or "").strip()
    if code: parts.append("[CODE]\n" + code)
    return "\n".join(parts)


# -------------- build --------------

def ensure_built(outdir: Path, embed_model: str) -> None:
    p = _resolve_paths(outdir)
    if p["index"].exists() and p["metas"].exists():
        return  # already built

    outdir.mkdir(parents=True, exist_ok=True)
    raw_docs = load_chunks_auto(outdir)
    docs = [normalize_chunk_keys(o, i) for i, o in enumerate(raw_docs)]

    texts = [build_text_for_embedding(d) for d in docs]
    client = _openai_client()

    # Embed in batches and L2-normalize
    B = 128
    vecs_list: List[np.ndarray] = []
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        arr = np.array([d.embedding for d in resp.data], dtype="float32")
        vecs_list.append(arr)
    vecs = np.vstack(vecs_list)
    vecs = _l2_normalize(vecs)

    # Save embedding.jsonl
    with p["embedding"].open("w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            f.write(json.dumps({"id": d.get("id"), "embedding": vecs[i].tolist()}, ensure_ascii=False) + "\n")

    # FAISS IP index
    faiss = _try_import_faiss()
    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(p["index"]))

    # metas.pkl
    with p["metas"].open("wb") as f:
        pickle.dump(docs, f)

    print(f"Built index in {outdir} (n={len(docs)}, dim={dim})")


# -------------- retrieval --------------

def retrieve(outdir: Path, query: str, topk: int, embed_model: str) -> List[Dict[str, Any]]:
    p = _resolve_paths(outdir)
    faiss = _try_import_faiss()
    index = faiss.read_index(str(p["index"]))
    metas: List[Dict[str, Any]] = pickle.loads(p["metas"].read_bytes())

    client = _openai_client()
    resp = client.embeddings.create(model=embed_model, input=[query])
    qvec = np.array([resp.data[0].embedding], dtype="float32")
    qvec = _l2_normalize(qvec)

    scores, I = index.search(qvec, topk)
    result: List[Dict[str, Any]] = []
    for idx in I[0]:
        if idx == -1: continue
        result.append(metas[int(idx)])
    return result


# -------------- prompting --------------
SECTION_LABEL = {
    "method": "Java Method",
    "function": "JavaScript Function",
    "sql": "SQL Mapper",
    "HTML_VIEW": "HTML View",
    "STYLE": "Style Block",
}

def _bucket_key(c: Dict[str, Any]) -> str:
    t = (c.get("type") or "").upper()
    if t in SECTION_LABEL: return t
    return c.get("type") or c.get("lang") or "unknown"


def build_context(chunks: List[Dict[str, Any]]) -> str:
    buckets: Dict[str, List[str]] = {}
    for c in chunks:
        key = _bucket_key(c)
        header = SECTION_LABEL.get(key, key)
        fp = c.get("file_path", ""); ident = c.get("id", "")
        doc = (c.get("docstring") or "").strip()
        code = (c.get("content") or "").strip()
        parts: List[str] = []
        if fp or ident: parts.append(f"[SRC] {fp} | {ident}")
        if doc: parts.append(f"[DOC] {doc}")
        if code: parts.append(f"[CODE]\n{code}")
        buckets.setdefault(header, []).append("\n".join(parts))

    lines: List[str] = []
    for header, items in buckets.items():
        lines.append(f"### {header}")
        for it in items:
            lines.append(it)
            lines.append("")
    return "\n".join(lines)


# -------------- generation --------------
FENCE_RE = re.compile(r"```(\w+)\n([\s\S]*?)```", re.IGNORECASE)
CLASS_RE = re.compile(r"class\s+(\w+)")
MAPPER_NS_RE = re.compile(r"<mapper[^>]*namespace=\"([^\"]+)\"", re.IGNORECASE)


def generate_code(system_prompt: str, context: str, task: str, chat_model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 1600) -> str:
    client = _openai_client()
    user_prompt = (
        "## Retrieved Context\n" + context + "\n\n" +
        "## Instruction\n" + task + "\n\n" +
        "## Output format (VERY IMPORTANT)\n"
        "Return four fenced code blocks in this order (omit block if not applicable):\n"
        "1) ```html\n...\n```\n"
        "2) ```java\n// Controller\n...\n```\n"
        "3) ```java\n// Service\n...\n```\n"
        "4) ```xml\n...\n```\n"
    )
    resp = client.chat.completions.create(
        model=chat_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content or ""


def _pick_java_blocks(blocks: List[str]) -> Tuple[Optional[str], Optional[str]]:
    controller = None; service = None
    for code in blocks:
        if re.search(r"@RestController|@Controller|@RequestMapping|@PostMapping|@GetMapping", code):
            controller = controller or code
        if re.search(r"@Service|class\s+\w+Service\b", code):
            service = service or code
    if not controller and blocks: controller = blocks[0]
    if not service and len(blocks) > 1: service = blocks[1]
    return controller, service


def extract_blocks(output: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    html = None; controller = None; service = None; mapper = None
    fences = FENCE_RE.findall(output)
    java_blocks: List[str] = []
    for lang, code in fences:
        lang_low = lang.lower()
        if lang_low == "html": html = code.strip()
        elif lang_low == "xml": mapper = code.strip()
        elif lang_low == "java": java_blocks.append(code.strip())
    if java_blocks:
        controller, service = _pick_java_blocks(java_blocks)
    return html, controller, service, mapper


def _class_name(java_src: str) -> Optional[str]:
    m = CLASS_RE.search(java_src or "");
    return m.group(1) if m else None


def _mapper_name(xml_src: str) -> str:
    m = MAPPER_NS_RE.search(xml_src or "")
    if m:
        base = m.group(1).split(".")[-1]
        return f"{base}.xml"
    return "Mapper.xml"


def save_files(outdir: Path, html: Optional[str], controller: Optional[str], service: Optional[str], mapper: Optional[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if html:
        (outdir / "NewSearch.html").write_text(html, encoding="utf-8")
    if controller:
        cname = _class_name(controller) or "Controller"
        (outdir / f"{cname}.java").write_text(controller, encoding="utf-8")
    if service:
        sname = _class_name(service) or "Service"
        (outdir / f"{sname}.java").write_text(service, encoding="utf-8")
    if mapper:
        mname = _mapper_name(mapper)
        (outdir / mname).write_text(mapper, encoding="utf-8")


# -------------- defaults & CLI --------------
DEFAULT_SYSTEM_PROMPT = (
    "당신은 우리 회사의 Java/Spring 전문가입니다.\n"
    "반드시 제공된 컨텍스트의 코드/SQL만을 근거로 답하세요.\n"
    "최종 답에는 사용한 근거의 [SRC id/file_path]를 인라인로 남기세요.\n"
    "프로젝트 아키텍처:\n"
    "- Frontend: HTML 템플릿 (search.html 스타일)\n"
    "- Controller: MainController.java /account 핸들링\n"
    "- Service: AccountService.execute(oper, payload) 분기\n"
    "- Mapper: AccountMapper.xml MyBatis 쿼리\n"
)
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


def main():
    ap = argparse.ArgumentParser(description="RAG (OpenAI-only) single-command")
    ap.add_argument("--query", required=True, help="What you want to generate")
    ap.add_argument("--outdir", default="../output")
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--chat-model", default=CHAT_MODEL)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # 1) build if needed
    ensure_built(outdir, args.embed_model)

    # 2) retrieve
    chunks = retrieve(outdir, args.query, args.topk, args.embed_model)

    # 3) context
    context = build_context(chunks)

    # 4) generate
    output = generate_code(DEFAULT_SYSTEM_PROMPT, context, args.query, chat_model=args.chat_model)

    # 5) save into ./output
    html, ctrl, svc, mapper = extract_blocks(output)
    save_files(outdir, html, ctrl, svc, mapper)

    print(f"Done. Files saved under {outdir}")


if __name__ == "__main__":
    main()
