# -*- coding: utf-8 -*-
"""Rebuild English policy sub-index for 3-step RAG.

1) Re-extract local PDFs; if cid/garbage, fetch Appendix A official URL.
2) Translate usable Chinese to English (gpt-5.4-mini, temp 0).
3) Chunk 512/128 and encode; save policy_en embeddings only.
"""
from __future__ import annotations

import glob
import html as htmlmod
import json
import os
import re
import sys
import time
from datetime import datetime
from urllib.parse import urljoin

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\PAPER1\revision\GPT\secrets")
from load_api_key import load_api_key

import numpy as np
import pdfplumber
import pymupdf
import requests
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

API_KEY = load_api_key()
BASE_URL = "https://api.ohmygpt.com/v1/chat/completions"
TRANS_MODEL = "gpt-5.4-mini"
POLICY_DIR = r"E:\justice assessment\downloaded_documents"
LINKS = r"E:\PAPER1\revision\data\appendix_a_policy_links.json"
OUT_DIR = r"E:\PAPER1\revision\results\ablation_3step\policy_en"
ZH_JSON = os.path.join(OUT_DIR, "policy_zh.json")
EN_JSON = os.path.join(OUT_DIR, "policy_en.json")
NPZ = os.path.join(OUT_DIR, "policy_en_512_128.npz")
LOG = os.path.join(OUT_DIR, "rebuild.log")
CHUNK = 512
OVERLAP = 128
HAN_OK = 0.25
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def log(msg: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def han_ratio(s: str) -> float:
    if not s:
        return 0.0
    h = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    return h / max(len(s), 1)


def cid_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = len(re.findall(r"\(cid:\d+\)", s))
    return n * 8 / max(len(s), 1)


def extract_pdf(path: str) -> tuple[str, str]:
    best, method = "", "none"
    try:
        with pdfplumber.open(path) as doc:
            t = "\n".join((p.extract_text() or "") for p in doc.pages)
        if han_ratio(t) > han_ratio(best):
            best, method = t, "pdfplumber"
    except Exception as e:
        log(f"  pdfplumber fail {os.path.basename(path)}: {e}")
    try:
        doc = pymupdf.open(path)
        t = "\n".join(p.get_text() for p in doc)
        if han_ratio(t) > han_ratio(best):
            best, method = t, "pymupdf"
    except Exception as e:
        log(f"  pymupdf fail {os.path.basename(path)}: {e}")
    return best, method


def normalize_url(u: str) -> str:
    u = (u or "").strip()
    if u.startswith("/"):
        u = u.lstrip("/")
    if u and not u.startswith("http"):
        u = "https://" + u
    return u


def html_to_text(raw: str) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = htmlmod.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_url(url: str) -> tuple[str, str]:
    url = normalize_url(url)
    if not url:
        return "", "no_url"
    try:
        r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            tmp = os.path.join(OUT_DIR, "_tmp_fetch.pdf")
            with open(tmp, "wb") as f:
                f.write(r.content)
            t, method = extract_pdf(tmp)
            return t, "fetch_pdf:" + method
        enc = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else r.apparent_encoding
        raw = r.content.decode(enc or "utf-8", errors="replace")
        return html_to_text(raw), "fetch_html"
    except Exception as e:
        return "", f"fetch_error:{type(e).__name__}:{e}"


def load_links():
    rows = json.load(open(LINKS, encoding="utf-8"))
    out = []
    for row in rows:
        if len(row) < 3:
            continue
        out.append({"zh_name": row[0], "en_name": row[1], "url": row[2]})
    return out


def match_link(filename: str, links: list[dict]) -> dict | None:
    stem = os.path.splitext(filename)[0]
    hits = []
    for L in links:
        name = L["zh_name"]
        key = re.sub(r"[《》〈〉（）()\[\]\s]", "", name)
        if key and key[:8] in stem:
            hits.append((len(key), L))
        elif name[:6] in stem:
            hits.append((len(name), L))
    if not hits:
        return None
    hits.sort(reverse=True)
    return hits[0][1]


def collect_zh() -> list[dict]:
    links = load_links()
    pdfs = glob.glob(os.path.join(POLICY_DIR, "*.pdf"))
    recs = []
    used_names = set()
    for path in sorted(pdfs):
        fn = os.path.basename(path)
        local, method = extract_pdf(path)
        rec = {
            "file": fn,
            "path": path,
            "zh_name": None,
            "url": None,
            "extract_method": method,
            "local_han": round(han_ratio(local), 3),
            "local_cid": round(cid_ratio(local), 3),
            "text": local,
            "source": "pdf",
        }
        link = match_link(fn, links)
        if link:
            rec["zh_name"] = link["zh_name"]
            rec["url"] = link["url"]
            rec["en_name"] = link["en_name"]
            used_names.add(link["zh_name"])
        usable = han_ratio(local) >= HAN_OK and cid_ratio(local) < 0.05
        if not usable and rec.get("url"):
            log(f"fetch {fn} -> {rec['url'][:70]}")
            t, how = fetch_url(rec["url"])
            rec["fetch_method"] = how
            rec["fetch_han"] = round(han_ratio(t), 3)
            rec["fetch_n"] = len(t)
            if han_ratio(t) > han_ratio(rec["text"]):
                rec["text"] = t
                rec["source"] = how
                rec["extract_method"] = how
        rec["final_han"] = round(han_ratio(rec["text"]), 3)
        rec["n_chars"] = len(rec["text"])
        rec["usable"] = rec["final_han"] >= HAN_OK
        log(
            f"{'OK' if rec['usable'] else 'DROP'} {fn[:40]} han={rec['final_han']} "
            f"n={rec['n_chars']} via={rec['extract_method']}"
        )
        recs.append(rec)
    # Appendix A docs with no matching PDF
    for L in links:
        if L["zh_name"] in used_names:
            continue
        log(f"appendix-only fetch {L['zh_name'][:24]} {L['url'][:50]}")
        t, how = fetch_url(L["url"])
        recs.append(
            {
                "file": None,
                "zh_name": L["zh_name"],
                "en_name": L["en_name"],
                "url": L["url"],
                "text": t,
                "source": how,
                "extract_method": how,
                "final_han": round(han_ratio(t), 3),
                "n_chars": len(t),
                "usable": han_ratio(t) >= HAN_OK,
            }
        )
        log(f"  han={han_ratio(t):.3f} n={len(t)} via={how}")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(recs, open(ZH_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_ok = sum(1 for r in recs if r["usable"])
    log(f"saved {ZH_JSON} usable={n_ok}/{len(recs)}")
    return recs


def split_zh(text: str, max_chars: int = 3500) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return [text]
    parts, buf = [], ""
    for para in re.split(r"(\n{2,})", text):
        if len(buf) + len(para) <= max_chars:
            buf += para
        else:
            if buf.strip():
                parts.append(buf.strip())
            if len(para) <= max_chars:
                buf = para
            else:
                for i in range(0, len(para), max_chars):
                    parts.append(para[i : i + max_chars])
                buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return [p for p in parts if len(p.strip()) > 20]


def translate_chunk(zh: str) -> str:
    prompt = (
        "Translate the following Chinese urban-planning / transport-policy text into clear English. "
        "Keep legal/article numbering. Do not summarize. Return English only.\n\n"
        + zh
    )
    payload = {
        "model": TRANS_MODEL,
        "temperature": 0.0,
        "max_completion_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    last = None
    for attempt in range(5):
        try:
            r = requests.post(
                BASE_URL,
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                timeout=180,
            )
            if r.status_code >= 400:
                raise RuntimeError(r.text[:300])
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise last


def translate_all(recs: list[dict]) -> list[dict]:
    usable = [r for r in recs if r.get("usable") and r.get("n_chars", 0) >= 800]
    usable.sort(key=lambda r: -r.get("n_chars", 0))
    picked = []
    seen_names = set()
    seen_lens = []
    for r in usable:
        name = r.get("zh_name")
        if name and name in seen_names:
            continue
        n = r.get("n_chars", 0)
        if any(abs(n - x) < 80 for x in seen_lens) and not name:
            continue
        if name:
            seen_names.add(name)
        seen_lens.append(n)
        picked.append(r)
    log(f"translate pool {len(picked)} docs (from {len(usable)} usable after dedupe)")
    recs = picked
    out = []
    cache_path = os.path.join(OUT_DIR, "translate_cache.jsonl")
    done_keys = set()
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done_keys.add(rec["key"])
                out.append(rec)
        log(f"translate cache {len(out)} docs")
    for rec in recs:
        if not rec.get("usable"):
            continue
        key = rec.get("zh_name") or rec.get("file") or str(len(out))
        if any(x.get("key") == key for x in out):
            log(f"skip translated {key}")
            continue
        chunks = split_zh(rec["text"])
        log(f"translate {key} chunks={len(chunks)} chars={rec['n_chars']}")
        en_parts = []
        for i, ch in enumerate(chunks, 1):
            try:
                en = translate_chunk(ch)
            except Exception as e:
                log(f"  {i}/{len(chunks)} SKIP {type(e).__name__}: {str(e)[:160]}")
                continue
            if not en or len(en) < 40:
                log(f"  {i}/{len(chunks)} empty, skip chunk")
                continue
            en_parts.append(en)
            log(f"  {i}/{len(chunks)} en_chars={len(en)}")
        if not en_parts:
            log(f"drop doc {key}: no English chunks")
            continue
        if len(en_parts) < len(chunks):
            log(f"  partial keep {len(en_parts)}/{len(chunks)} chunks")
        item = {
            "key": key,
            "zh_name": rec.get("zh_name"),
            "en_name": rec.get("en_name"),
            "file": rec.get("file"),
            "url": rec.get("url"),
            "source": rec.get("source"),
            "n_zh": rec["n_chars"],
            "n_en_chunks": len(en_parts),
            "n_zh_chunks": len(chunks),
            "text": "\n\n".join(en_parts),
        }
        out.append(item)
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        json.dump(out, open(EN_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        log(f"saved {EN_JSON} n={len(out)}")
    log(f"translate done n={len(out)}")
    return out


def chunk_docs(docs: list[dict]) -> list[dict]:
    chunks = []
    for d in docs:
        toks = d["text"].split()
        for i in range(0, len(toks), CHUNK - OVERLAP):
            c = " ".join(toks[i : i + CHUNK])
            if len(c.split()) > 20:
                chunks.append(
                    {
                        "source": "policy",
                        "key": d.get("key"),
                        "zh_name": d.get("zh_name"),
                        "en_name": d.get("en_name"),
                        "text": c,
                    }
                )
    return chunks


def build_index(docs: list[dict]) -> None:
    chunks = chunk_docs(docs)
    log(f"policy-en chunks {len(chunks)}")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    np.savez_compressed(
        NPZ,
        emb=emb,
        texts=np.array(texts, dtype=object),
        keys=np.array([c.get("key") or "" for c in chunks], dtype=object),
        zh_names=np.array([c.get("zh_name") or "" for c in chunks], dtype=object),
        en_names=np.array([c.get("en_name") or "" for c in chunks], dtype=object),
    )
    log(f"saved {NPZ}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extract", "translate", "index", "all"], default="all")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage in ("extract", "all"):
        recs = collect_zh()
    else:
        recs = json.load(open(ZH_JSON, encoding="utf-8"))
    if args.stage == "extract":
        return
    if args.stage in ("translate", "all"):
        en = translate_all(recs)
    else:
        en = json.load(open(EN_JSON, encoding="utf-8"))
    if args.stage == "translate":
        return
    if not en:
        raise SystemExit("no English policy docs to index")
    build_index(en)


if __name__ == "__main__":
    main()
