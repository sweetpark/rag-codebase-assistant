# Re-chunk the user's uploaded project into 200–400 token chunks with rich metadata
import os, re, json, math, hashlib, textwrap, pathlib
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from caas_jupyter_tools import display_dataframe_to_user

BASE = Path("/mnt/data")

# Discover files
paths = {
    "baseTx": BASE / "baseTx.java",
    "DemoAiApplication": BASE / "DemoAiApplication.java",
    "MainController": BASE / "MainController.java",
    "AccountService": BASE / "AccountService.java",
    "AccountRepository": BASE / "AccountRepository.java",
    "reg_html": BASE / "reg.html",
    "search_html": BASE / "search.html",
    "mapper": BASE / "AccountMapper.xml",
}

exists = {k: p.exists() for k,p in paths.items()}

# Helper: naive token estimate (~4 chars per token)
def toklen(s: str) -> int:
    return math.ceil(len(s)/4)

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except:
        return ""

# Extract Java methods (very heuristic but ok for demo)
method_pat = re.compile(r"(?:public|protected|private|\s) [\w\<\>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{", re.MULTILINE)

def split_java_methods(code: str) -> List[Dict[str, Any]]:
    blocks = []
    # Attempt to find method starts and brace-balance to end
    for m in method_pat.finditer(code):
        name = m.group(1)
        start = m.start()
        # naive brace matching from this point
        i = code.find("{", start)
        if i == -1: 
            continue
        depth = 1
        j = i+1
        while j < len(code) and depth > 0:
            if code[j] == "{": depth += 1
            elif code[j] == "}": depth -= 1
            j += 1
        method_src = code[start:j]
        blocks.append({"name": name, "src": method_src, "start": start, "end": j})
    return blocks

def extract_java_docstrings(code: str, block_start: int) -> str:
    # look backward for /** ... */
    pre = code[:block_start]
    m = re.search(r"/\*\*([\s\S]*?)\*/\s*$", pre)
    return m.group(1).strip() if m else ""

def java_chunks(file_key: str, path: Path) -> List[Dict[str, Any]]:
    code = read_text(path)
    if not code:
        return []
    blocks = split_java_methods(code)
    pkg_m = re.search(r"package\s+([a-zA-Z0-9_.]+);", code)
    pkg = pkg_m.group(1) if pkg_m else ""
    cls_m = re.search(r"class\s+(\w+)", code)
    cls = cls_m.group(1) if cls_m else path.stem
    out = []
    for b in blocks:
        doc = extract_java_docstrings(code, b["start"])
        content = b["src"]
        out.append({
            "id": f"{pkg}.{cls}.{b['name']}" if pkg else f"{cls}.{b['name']}",
            "file_path": str(path),
            "lang": "java",
            "type": "method",
            "content": content,
            "docstring": doc,
            "dependencies": [],
            "tags": ["java", cls],
        })
    # also create a class-level chunk with imports and mappings
    header = "\n".join(code.splitlines()[: min(120, len(code.splitlines())) ])
    out.append({
        "id": f"{pkg}.{cls}.__class__" if pkg else f"{cls}.__class__",
        "file_path": str(path),
        "lang": "java",
        "type": "class",
        "content": header,
        "docstring": "",
        "dependencies": [],
        "tags": ["java", cls, "class-header"],
    })
    return out

# Extract oper/http mapping from JS in HTML
def html_js_chunks(path: Path) -> List[Dict[str, Any]]:
    txt = read_text(path)
    if not txt:
        return []
    # find sendAjax calls and oper keys
    sendajax_calls = re.findall(r"sendAjax\(\s*'?(GET|POST|PUT|DELETE)'?\s*,\s*([\"'])?([a-zA-Z0-9_/-]+)\2?\s*,", txt)
    opers = set(re.findall(r"oper['\"]?\s*[:=]\s*['\"]([a-zA-Z0-9_]+)['\"]", txt))
    http_paths = set(re.findall(r"@PostMapping\(\s*['\"]([^'\"]+)['\"]\s*\)", txt))
    buttons = set(re.findall(r"data-oper=['\"]([a-zA-Z0-9_]+)['\"]", txt))
    meta = {
        "sendajax_methods": list({m for m,_,_ in sendajax_calls}),
        "sendajax_services": list({s for _,_,s in sendajax_calls}),
        "opers": list(opers | buttons),
        "http_paths": list(http_paths),
    }
    return [{
        "id": f"{path.name}.__view__",
        "file_path": str(path),
        "lang": "html" if path.suffix==".html" else "jsp",
        "type": "HTML_VIEW",
        "content": txt,
        "docstring": "",
        "dependencies": [],
        "tags": ["view", path.name] + meta.get("opers", []),
        "view_meta": meta,
    }]

# Extract MyBatis statements
def mapper_chunks(path: Path) -> List[Dict[str, Any]]:
    xml = read_text(path)
    if not xml:
        return []
    chunks = []
    for tag in ("select","insert","update","delete"):
        pattern = re.compile(rf"<{tag}\s+id=\"([^\"]+)\"[^>]*>([\s\S]*?)</{tag}>", re.MULTILINE)
        for m in pattern.finditer(xml):
            stmt_id, body = m.group(1), m.group(2).strip()
            chunks.append({
                "id": f"{path.stem}.{stmt_id}",
                "file_path": str(path),
                "lang": "xml",
                "type": "sql",
                "content": body,
                "docstring": "",
                "dependencies": [],
                "tags": ["mybatis", tag, stmt_id],
                "mapper_id": stmt_id
            })
    return chunks

# Build initial fine-grained chunks
chunks = []
chunks += java_chunks("baseTx", paths["baseTx"]) if exists["baseTx"] else []
chunks += java_chunks("DemoAiApplication", paths["DemoAiApplication"]) if exists["DemoAiApplication"] else []
chunks += java_chunks("MainController", paths["MainController"]) if exists["MainController"] else []
chunks += java_chunks("AccountService", paths["AccountService"]) if exists["AccountService"] else []
chunks += java_chunks("AccountRepository", paths["AccountRepository"]) if exists["AccountRepository"] else []
chunks += html_js_chunks(paths["reg_html"]) if exists["reg_html"] else []
chunks += html_js_chunks(paths["search_html"]) if exists["search_html"] else []
chunks += mapper_chunks(paths["mapper"]) if exists["mapper"] else []

# Build "flow" chunks: View → Controller → Service → Mapper, inferred by keywords
def infer_flows(chunks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # collect signals
    view_ops = {}
    for c in chunks:
        if c["type"]=="HTML_VIEW":
            ops = set(c.get("view_meta",{}).get("opers",[]))
            view_ops[c["file_path"]] = ops
    mapper_ids = {c.get("mapper_id"): c for c in chunks if c["type"]=="sql"}
    # Service methods mapping by keywords
    service_methods = [c for c in chunks if c["lang"]=="java" and "AccountService" in " ".join(c.get("tags",[])) or "AccountService" in c["id"]]
    service_text_map = {c["id"]: c["content"] for c in service_methods}
    flows = []
    for vpath, ops in view_ops.items():
        for op in ops:
            # guess mapper id candidates (dupCheck/read/readOne/reset/regist/edit etc. → map to known mapper ids)
            candidates = [k for k in mapper_ids.keys() if k.lower().startswith(op.lower()) or op.lower() in k.lower()]
            service_hits = [sid for sid,src in service_text_map.items() if op in src]
            flow_text = f"""[FLOW] oper="{op}"\n\n- View: {Path(vpath).name}\n- Controller: MainController (guess)\n- Service methods: {service_hits or 'N/A'}\n- Mapper candidates: {candidates or 'N/A'}\n"""
            flows.append({
                "id": f"flow::{Path(vpath).stem}::{op}",
                "file_path": str(vpath),
                "lang": "txt",
                "type": "flow",
                "content": flow_text,
                "docstring": "View→Controller→Service→Mapper 추정 경로",
                "dependencies": service_hits + candidates,
                "tags": ["flow", op]
            })
    return flows

chunks += infer_flows(chunks)

# Normalize chunks to ~200–400 tokens with soft overlap
TARGET_MIN, TARGET_MAX = 200, 400
OVERLAP_TOK = 40

def split_long_content(base_chunk: Dict[str,Any]) -> List[Dict[str,Any]]:
    txt = base_chunk["content"]
    tokens = toklen(txt)
    if tokens <= TARGET_MAX:
        return [base_chunk]
    # split by lines to preserve code
    lines = txt.splitlines()
    pieces = []
    cur = []
    cur_tok = 0
    part_idx = 1
    for line in lines:
        t = toklen(line+"\n")
        if cur_tok + t > TARGET_MAX and cur_tok >= TARGET_MIN:
            # flush
            sub = "\n".join(cur)
            piece = base_chunk.copy()
            piece["id"] = f"{base_chunk['id']}#part{part_idx}"
            piece["content"] = sub
            piece["part"] = part_idx
            pieces.append(piece)
            # start new with overlap
            overlap = []
            tok_acc = 0
            for l in reversed(cur):
                tok_acc += toklen(l+"\n")
                overlap.append(l)
                if tok_acc >= OVERLAP_TOK:
                    break
            cur = list(reversed(overlap))
            cur_tok = sum(toklen(l+"\n") for l in cur)
            part_idx += 1
        cur.append(line)
        cur_tok += t
    # flush last
    if cur:
        sub = "\n".join(cur)
        piece = base_chunk.copy()
        piece["id"] = f"{base_chunk['id']}#part{part_idx}"
        piece["content"] = sub
        piece["part"] = part_idx
        pieces.append(piece)
    return pieces

# Apply splitting
final_chunks = []
for ch in chunks:
    final_chunks.extend(split_long_content(ch))

# Add computed fields
for ch in final_chunks:
    ch["content_tokens_est"] = toklen(ch["content"])
    ch["module"] = "demo_ai"
    # simple hash
    ch["uid"] = hashlib.sha1((ch["id"] + "|" + ch["file_path"]).encode("utf-8")).hexdigest()

# Save JSONL
out_path = BASE / "rechunk.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for ch in final_chunks:
        f.write(json.dumps(ch, ensure_ascii=False) + "\n")

# Show a small preview
preview_cols = ["id","type","lang","file_path","content_tokens_est","tags"]
preview_df = pd.DataFrame(final_chunks)[preview_cols].head(20)
display_dataframe_to_user("Rechunk preview", preview_df)

out_path, len(final_chunks), exists
