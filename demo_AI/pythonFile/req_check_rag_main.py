import argparse
import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ------------------------------- OpenAI client -------------------------------

def _openai_client():
    try:
        from openai import OpenAI  # new SDK
    except Exception as e:
        raise RuntimeError("Install openai>=1.30 (`pip install openai`)") from e
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI()


# --------------------------------- FAISS -------------------------------------

def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception as e:
        raise RuntimeError("Install faiss-cpu (`pip install faiss-cpu`)") from e


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


# ------------------------------ Path resolution ------------------------------

DEFAULT_CHUNK_FILES = [
    "new_chunk.jsonl", "new_chunk.json",
    "rechunk_noisy.jsonl", "rechunk_noisy.json",
    "chunk.jsonl", "chunk.json"
]

def _resolve_paths(outdir: Path) -> Dict[str, Path]:
    p = {
        "outdir": outdir,
        "index": outdir / "code_index.faiss",
        "metas": outdir / "metas.pkl",
        "metrics": outdir / "metrics.jsonl",
    }
    for name in DEFAULT_CHUNK_FILES:
        if (outdir / name).exists():
            p["chunks"] = outdir / name
            break
    else:
        p["chunks"] = outdir / DEFAULT_CHUNK_FILES[0]
    return p


def _load_jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    txt = path.read_text(encoding="utf-8").strip()
    docs: List[Dict[str, Any]] = []
    if txt.startswith("["):
        try:
            arr = json.loads(txt)
            if isinstance(arr, list):
                return [dict(o) for o in arr]
        except Exception:
            pass
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            docs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return docs


def normalize_chunk_keys(obj: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Normalize keys from various chunkers into a common schema."""
    o = dict(obj)
    if "id" not in o or not o["id"]:
        o["id"] = o.get("chunk_id") or o.get("uuid") or f"chunk_{idx}"
    o["content"] = o.get("text") or o.get("content") or o.get("code") or ""
    if "file_path" not in o:
        o["file_path"] = o.get("path") or o.get("source") or o.get("file") or ""
    o["lang"] = o.get("lang") or o.get("language") or o.get("ext") or ""
    o["type"] = o.get("type") or o.get("kind") or ""
    o["docstring"] = o.get("docstring") or o.get("description") or ""
    o["tags"] = o.get("tags") or []
    return o


def load_chunks_auto(outdir: Path) -> List[Dict[str, Any]]:
    p = _resolve_paths(outdir)
    docs = _load_jsonl_or_json(p["chunks"])
    return [normalize_chunk_keys(d, i) for i, d in enumerate(docs)]


# --------------------------- Embedding text minimal ---------------------------

def build_text_for_embedding(item: Dict[str, Any]) -> str:
    """Minimal header + body to avoid vector pollution by tags/commit/etc."""
    t = (item.get("type") or "").upper()
    l = (item.get("lang") or "").lower()
    fp = item.get("file_path") or ""
    body = (item.get("content") or "").strip()
    doc = (item.get("docstring") or "").strip() if t in {"METHOD", "FUNCTION", "SQL"} else ""

    parts: List[str] = []
    if fp: parts.append(f"[FILE] {fp}")
    if t:  parts.append(f"[TYPE] {t}")
    if l:  parts.append(f"[LANG] {l}")
    if doc: parts.append("[DOC]\n" + doc)
    if body: parts.append("[BODY]\n" + body)
    return "\n".join(parts)


# ---------------------------------- Build ------------------------------------

def _embed_batch(texts: List[str], model: str) -> np.ndarray:
    client = _openai_client()
    B = 64
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), B):
        sub = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=sub)
        vecs.extend([np.array(e.embedding, dtype="float32") for e in resp.data])
    X = np.vstack(vecs) if vecs else np.zeros((0, 1536), dtype="float32")
    return _l2_normalize(X)


def ensure_built(outdir: Path, embed_model: str) -> Tuple[int, int]:
    """If FAISS/metas not found, build them. Returns: (num_chunks, dim)."""
    faiss = _try_import_faiss()
    p = _resolve_paths(outdir)
    if p["index"].exists() and p["metas"].exists():
        index = faiss.read_index(str(p["index"]))
        return (len(pickle.loads(p["metas"].read_bytes())), index.d)
    outdir.mkdir(parents=True, exist_ok=True)
    raw = load_chunks_auto(outdir)
    metas = [normalize_chunk_keys(o, i) for i, o in enumerate(raw)]
    texts = [build_text_for_embedding(m) for m in metas]
    X = _embed_batch(texts, embed_model)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(X)
    faiss.write_index(index, str(p["index"]))
    p["metas"].write_bytes(pickle.dumps(metas))
    return (len(metas), dim)


# -------------------------------- Retrieval ----------------------------------

def retrieve_with_scores(outdir: Path, query: str, topk: int, embed_model: str, fetch_k: Optional[int] = None
                         ) -> Tuple[List[Dict[str, Any]], List[int], List[float], Dict[str, int], int]:
    """
    returns: (metas_topk, index_ids, scores, usage, latency_ms)
    Note: fetch_k lets us overfetch for later type-minima enforcement.
    """
    try:
        topk = int(topk)
    except Exception:
        topk = 20
    if fetch_k is None:
        fetch_k = max(60, topk)  # sensible default overfetch
    else:
        fetch_k = max(int(fetch_k), topk, 1)

    faiss = _try_import_faiss()
    p = _resolve_paths(outdir)
    index = faiss.read_index(str(p["index"]))
    metas: List[Dict[str, Any]] = pickle.loads(p["metas"].read_bytes())

    t0 = int(time.time() * 1000)
    client = _openai_client()
    resp = client.embeddings.create(model=embed_model, input=[query])
    qvec = np.array([resp.data[0].embedding], dtype="float32")
    qvec = _l2_normalize(qvec)
    scores, I = index.search(qvec, fetch_k)
    ids = I[0].tolist()
    sc = scores[0].tolist()
    items: List[Dict[str, Any]] = []
    for idx in ids:
        if idx == -1: continue
        items.append(metas[int(idx)])
    u = getattr(resp, "usage", None)
    usage = {
        "embed_prompt_tokens": getattr(u, "prompt_tokens", 0) if u else 0,
        "embed_total_tokens": getattr(u, "total_tokens", 0) if u else 0,
    }
    t1 = int(time.time() * 1000)
    return items, ids, sc, usage, (t1 - t0)


# ---------------- require-types (type minima) utilities -----------------------

_CTRL_RE = re.compile(r"@RestController|@Controller|@RequestMapping|@GetMapping|@PostMapping|@PutMapping|@PatchMapping|@DeleteMapping", re.IGNORECASE)
_SVC_RE  = re.compile(r"@Service|class\s+\w+Service\b", re.IGNORECASE)

def parse_require_types(s: str) -> Dict[str, int]:
    """
    "HTML_VIEW:1,METHOD_CONTROLLER:1,METHOD_SERVICE:1,SQL:1" -> dict
    """
    req: Dict[str, int] = {}
    if not s:
        return req
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            t, n = part.split(":", 1)
            try:
                req[t.strip().upper()] = max(0, int(n))
            except:
                continue
        else:
            req[part.strip().upper()] = 1
    return req

def _method_role(meta: dict) -> str:
    """
    Java METHOD 청크에서 Controller/Service 구분.
    """
    t = (meta.get("type") or "").upper()
    if t != "METHOD":
        return t
    body = (meta.get("content") or "")
    if _CTRL_RE.search(body):
        return "METHOD_CONTROLLER"
    if _SVC_RE.search(body):
        return "METHOD_SERVICE"
    return "METHOD"

def _bucket_key_for_minima(meta: dict) -> str:
    t = (meta.get("type") or "").upper()
    if t == "METHOD":
        return _method_role(meta)
    return t

from typing import Iterable
def enforce_type_minima(
    candidates: List[Tuple[int, float, dict]],
    topk: int,
    minima: Dict[str, int]
) -> List[Tuple[int, float, dict]]:
    """
    candidates: [(faiss_idx, score, meta)], any order.
    1) sort by score desc
    2) satisfy minima per bucket
    3) fill remaining slots by score desc
    """
    if not candidates or topk <= 0:
        return []
    cands_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

    # bucketize
    buckets: Dict[str, List[Tuple[int, float, dict]]] = {}
    for it in cands_sorted:
        key = _bucket_key_for_minima(it[2])
        buckets.setdefault(key, []).append(it)

    selected: List[Tuple[int, float, dict]] = []
    used = set()

    # satisfy minima
    for key, need in minima.items():
        bucket = buckets.get(key, [])
        take = bucket[:need]
        for it in take:
            if it[0] in used:
                continue
            selected.append(it)
            used.add(it[0])

    # fill remaining
    for it in cands_sorted:
        if len(selected) >= topk:
            break
        if it[0] in used:
            continue
        selected.append(it)
        used.add(it[0])

    return selected[:topk]


# -------------------------------- Context ------------------------------------

SECTION_LABEL = {
    "class": "Java Method",
    "controller": "Controller (Java)",
    "base":"Base Tx Interface",
    "service": "Service (Java)",
    "FUNCTION": "JavaScript Function",
    "sql": "SQL Mapper",
    "HTML_VIEW": "HTML View",
    "STYLE": "Style Block",
}

def _bucket_key(c: Dict[str, Any]) -> str:
    t = (c.get("type") or "").upper()
    if t == "METHOD":
        return _method_role(c)
    if t in SECTION_LABEL: return t
    return (c.get("type") or c.get("lang") or "unknown").upper()


def build_context(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for c in chunks:
        key = _bucket_key(c)
        header = SECTION_LABEL.get(key, key)
        fp = c.get("file_path", "")
        cid = c.get("id", "")
        doc = (c.get("docstring") or "").strip()
        body = (c.get("content") or "").strip()
        lines.append(f"### {header} | [SRC id={cid} file_path={fp}]")
        if doc:
            lines.append(doc)
        if body:
            lines.append("```")
            lines.append(body)
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


# -------------------------------- Generate -----------------------------------

FENCE_RE = re.compile(r"```(\w+)\n([\s\S]*?)```", re.IGNORECASE)
CLASS_RE = re.compile(r"class\s+(\w+)")
MAPPER_NS_RE = re.compile(r"<mapper[^>]*namespace=\"([^\"]+)\"", re.IGNORECASE)

DEFAULT_SYSTEM_PROMPT = (
    "당신은 우리 회사의 Java/Spring 전문가입니다.\n"
    "반드시 제공된 컨텍스트의 코드/SQL만을 근거로 답하세요.\n"
    "컨텍스트 밖 추론을 하지 말고, 사용한 근거의 [SRC id/file_path]를 인라인으로 남기세요.\n"
    "프로젝트 아키텍처:\n"
    "- Frontend: HTML 템플릿 (search.html 스타일)\n"
    "- Controller: MainController.java /account 핸들링\n"
    "- Service: AccountService.execute(oper, payload) 분기\n"
    "- Mapper: AccountMapper.xml MyBatis 쿼리\n"
)

@dataclass
class GenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def generate_code(system_prompt: str, context: str, task: str,
                  chat_model: str = "gpt-4o-mini",
                  temperature: float = 0.2, max_tokens: int = 1600) -> Tuple[str, GenUsage]:
    client = _openai_client()
    user_prompt = (
        "## Retrieved Context\n" + context + "\n\n" +
        "## Instruction\n" + task + "\n\n" +
        "## Output format (STRICT)\n"
        "Return up to four fenced code blocks in this order (omit if not applicable):\n"
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
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    usage = GenUsage(
        prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
        completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        total_tokens=getattr(u, "total_tokens", 0) if u else 0,
    )
    return content, usage


def _class_name(code: str) -> Optional[str]:
    m = CLASS_RE.search(code or "")
    return m.group(1) if m else None


def _mapper_name(xml: str) -> str:
    ns = "Mapper.xml"
    m = MAPPER_NS_RE.search(xml or "")
    if m:
        ns = m.group(1).split(".")[-1] + ".xml"
    return ns


def _pick_java_blocks(blocks: List[str]) -> Tuple[Optional[str], Optional[str]]:
    controller = None; service = None
    for code in blocks:
        if re.search(r"@RestController|@Controller|@RequestMapping|@PostMapping|@GetMapping", code or ""):
            controller = controller or code
        if re.search(r"@Service|class\s+\w+Service\b", code or ""):
            service = service or code
    if not controller and blocks: controller = blocks[0]
    if not service and len(blocks) > 1: service = blocks[1]
    return controller, service


def extract_blocks(output: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    html = None; controller = None; service = None; mapper = None
    fences = FENCE_RE.findall(output or "")
    java_blocks: List[str] = []
    for lang, code in fences:
        lang_low = (lang or "").lower()
        code = (code or "").strip()
        if not code:
            continue
        if lang_low == "html":
            html = code
        elif lang_low == "xml":
            mapper = code
        elif lang_low == "java":
            java_blocks.append(code)
    if java_blocks:
        controller, service = _pick_java_blocks(java_blocks)
    return html, controller, service, mapper


def save_files(outdir: Path,
               html: Optional[str],
               controller: Optional[str],
               service: Optional[str],
               mapper: Optional[str]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    if html:
        (outdir / "NewSearch.html").write_text(html, encoding="utf-8")
    if controller:
        cname = _class_name(controller) or "MainController"
        (outdir / f"{cname}.java").write_text(controller, encoding="utf-8")
    if service:
        sname = _class_name(service) or "AccountService"
        (outdir / f"{sname}.java").write_text(service, encoding="utf-8")
    if mapper:
        mname = _mapper_name(mapper)
        (outdir / mname).write_text(mapper, encoding="utf-8")


# ------------------------------ Evaluation utils -----------------------------

SRC_RE = re.compile(r"\[SRC[^\]]*\]", re.IGNORECASE)

def extract_citations(text: str) -> List[str]:
    return [m.group(0) for m in SRC_RE.finditer(text or "")]

def citations_in_context(citations: List[str], retrieved: List[Dict[str, Any]]) -> Tuple[int, int]:
    if not citations:
        return (0, 0)
    keys: List[str] = []
    for c in retrieved:
        if c.get("id"): keys.append(str(c["id"]))
        if c.get("file_path"): keys.append(str(c["file_path"]))
    in_ctx = 0
    for cit in citations:
        if any(k and k in cit for k in keys):
            in_ctx += 1
    return in_ctx, len(citations)


def compute_retrieval_metrics(retrieved_ids: List[int],
                              metas: List[Dict[str, Any]],
                              relevant_ids: Optional[List[str]] = None,
                              k: int = 10) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    id_list = [metas[i].get("id") if i != -1 else None for i in retrieved_ids]
    res["retrieved_chunk_ids"] = id_list[:k]
    first_hit_rank = None
    if relevant_ids:
        target = set(relevant_ids)
        for r, cid in enumerate(id_list, start=1):
            if cid and cid in target:
                first_hit_rank = r
                break
    res["first_hit_rank"] = first_hit_rank
    res["mrr"] = (1.0 / first_hit_rank) if first_hit_rank else 0.0
    if relevant_ids:
        rel = set(relevant_ids)
        hit = [cid for cid in id_list[:k] if cid in rel]
        res["precision_at_k"] = (len(hit) / float(k)) if k else 0.0
        res["recall_at_k"] = (len(hit) / float(len(rel))) if rel else 0.0
        res["hits_at_k"] = len(hit)
        res["relevant_count"] = len(rel)
    return res


def compute_generation_metrics(output: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:
    html, controller, service, mapper = extract_blocks(output or "")
    cits = extract_citations(output or "")
    in_ctx, total = citations_in_context(cits, retrieved or [])
    return {
        "has_html": bool(html),
        "has_controller": bool(controller),
        "has_service": bool(service),
        "has_mapper": bool(mapper),
        "citations_total": total,
        "citations_in_context": in_ctx,
        "citation_in_context_rate": (in_ctx / total) if total else None,
    }


def save_metrics(outdir: Path, metrics: Dict[str, Any], metrics_filename: str = "metrics.jsonl") -> None:
    p = outdir / metrics_filename
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


# ----------------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="RAG (OpenAI + FAISS) with type-minima enforcement")
    ap.add_argument("--query", required=True, help="The task/question")
    ap.add_argument("--outdir", default="../output")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--chat-model", default="gpt-4o-mini")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--overfetch", type=int, default=60, help="Retrieve more than topk for minima enforcement")
    ap.add_argument("--require-types", default="HTML_VIEW:1,service:1,base:1,method:1,sql:1, flow:1, controller:1",
                    help="type:min pairs, comma-separated. e.g., HTML_VIEW:1,METHOD:2,SQL:1")
    ap.add_argument("--relevant-ids", default="", help="comma-separated relevant chunk ids for evaluation (optional)")
    ap.add_argument("--metrics-out", default="metrics.jsonl", help="metrics output file name under outdir")
    args = ap.parse_args()

    outdir = Path(args.outdir)

    # 0) Build (if needed)
    t_build0 = int(time.time() * 1000)
    n_chunks, dim = ensure_built(outdir, args.embed_model)
    t_build1 = int(time.time() * 1000)

    # 1) Retrieve with overfetch
    items, faiss_ids, scores, emb_usage, t_retrieval = retrieve_with_scores(
        outdir, args.query, args.topk, args.embed_model, fetch_k=args.overfetch
    )

    # Build candidate list (faiss_idx, score, meta)
    candidates = [(idx, float(sc), meta) for idx, sc, meta in zip(faiss_ids, scores, items)]

    # 2) Enforce type minima
    minima = parse_require_types(args.require_types)
    chosen = enforce_type_minima(candidates, args.topk, minima)

    # Decompose chosen
    chosen_items = [m for (_, _, m) in chosen]
    chosen_ids   = [int(i) for (i, _, _) in chosen]
    chosen_scores= [float(s) for (_, s, _) in chosen]

    # 3) Context
    context = build_context(chosen_items)

    # 4) Generate
    t_gen0 = int(time.time() * 1000)
    output, gen_usage = generate_code(DEFAULT_SYSTEM_PROMPT, context, args.query, chat_model=args.chat_model)
    t_gen1 = int(time.time() * 1000)

    # 5) Save files
    html, ctrl, svc, mapper = extract_blocks(output)
    save_files(outdir, html, ctrl, svc, mapper)

    # 6) Metrics
    p = _resolve_paths(outdir)
    metas: List[Dict[str, Any]] = pickle.loads(p["metas"].read_bytes())
    relevant_ids = [s.strip() for s in (args.relevant_ids or "").split(",") if s.strip()]

    ret_metrics = compute_retrieval_metrics(chosen_ids, metas, relevant_ids or None, k=args.topk)
    gen_metrics = compute_generation_metrics(output, chosen_items)

    metrics = {
        "query": args.query,
        "topk": args.topk,
        "overfetch": args.overfetch,
        "require_types": args.require_types,
        "embed_model": args.embed_model,
        "chat_model": args.chat_model,
        "index_dim": dim,
        "corpus_size": n_chunks,
        "latency_ms": {
            "build": (t_build1 - t_build0),
            "retrieve": t_retrieval,
            "generate": (t_gen1 - t_gen0),
            "total": (t_gen1 - t_build0),
        },
        "tokens": {
            "embedding_prompt": emb_usage.get("embed_prompt_tokens", 0),
            "embedding_total": emb_usage.get("embed_total_tokens", 0),
            "gen_prompt": gen_usage.prompt_tokens,
            "gen_completion": gen_usage.completion_tokens,
            "gen_total": gen_usage.total_tokens,
        },
        "retrieval": ret_metrics,
        "generation": gen_metrics,
    }

    save_metrics(outdir, metrics, args.metrics_out)

    print(f"[OK] Files saved under: {outdir.resolve()}")
    print(f"[OK] Metrics appended to: {(outdir/args.metrics_out).resolve()}")


if __name__ == "__main__":
    main()