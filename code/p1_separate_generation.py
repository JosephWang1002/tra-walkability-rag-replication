# -*- coding: utf-8 -*-
"""p1_separate_generation.py —— 共同生成 vs 分开生成对比实验（回应循环性质疑）
共同生成组：六维 + walkability 一次调用；分开生成组：六维和 walkability 分两次调用。
prompt 主体结构一致，仅「评估维度+JSON字段」不同。均 nano + full RAG。
"""
import os, sys, glob, json, time, base64
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'E:\PAPER1\revision\GPT\secrets')
from load_api_key import load_api_key
API_KEY = load_api_key()
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MODEL = "gpt-5.4-nano"
BASE_URL = "https://api.ohmygpt.com/v1/chat/completions"
IMG_DIR = r'e:\PAPER1\human_sampled_images_200'
CACHE = r'E:\PAPER1\revision\data\corpus_with_source.json'
OUT_RAW = r'e:\PAPER1\revision\results\ablation\separate_generation_raw.json'
OUT_DIR = r'e:\PAPER1\revision\results\ablation'
N = 200
SEED = 42
TEMPERATURE = 0.0

DIMS6 = ["justice","safety","livability","sustainability","aesthetics","comfort"]
DIMS_ALL = DIMS6 + ["walkability"]
PERSONA = 'all_all'
QUERY = f'streetscape guidelines for {PERSONA} related to {", ".join(DIMS_ALL)}'
TOP_K = 8
CHUNK = 512
OVERLAP = 128

def build_prompt(dims, context):
    indicators = ", ".join(dims)
    rag = '\n'.join(f'- [Source: unknown] {c}' for c in context)[:6000] if context else '(no retrieved reference)'
    json_scores = ", ".join(f'"{d}": 0' for d in dims)
    json_reasons = ", ".join(f'"{d}": ["reason_1", "reason_2", "reason_3"]' for d in dims)
    return f"""You are a professional urban street landscape and transportation planning evaluation expert, strictly adhering to international transportation engineering and urban design academic standards for assessment work.
Evaluation Basic Information:
Target evaluation subject: {PERSONA} (must prioritize meeting this group's core travel needs for accessibility, safety, and comfort)
Evaluation dimensions: {indicators}
Authoritative reference basis (transportation field academic literature + design specifications): {rag}
Evaluation data source: uploaded street scene images
Strictly Follow These Evaluation Rules:
Scoring standard: output a quantitative score between 0-100 for each evaluation dimension, with higher scores indicating better performance;
Reason requirements: each dimension must provide 3-5 core reasons that simultaneously meet two criteria: ① based on authentic visual features from the images; ② aligned with academic conclusions/design specifications in the reference basis;
Prohibited actions: forbidden fabricating information or generating content detached from the images and reference basis;
Output format: return only standard JSON format without any additional text, explanations, or comments.
Please Strictly Follow This JSON Structure for Output:
{{"scores": {{{json_scores}}}, "explanations": {{{json_reasons}}}}}"""

def call_llm(b64, prompt):
    payload = {"model": MODEL, "temperature": TEMPERATURE, "max_completion_tokens": 3000,
               "messages": [{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}]}
    last = None
    for attempt in range(6):
        try:
            r = requests.post(BASE_URL, json=payload, headers={"Authorization":f"Bearer {API_KEY}","Content-Type":"application/json"}, timeout=180)
            r.raise_for_status()
            return r.json()['choices'][0]['message'].get('content') or ''
        except Exception as e:
            last = e
            time.sleep(5 * (attempt + 1))
    raise last

def parse(c):
    c = c.strip()
    if c.startswith('```'): c = c.strip('`').replace('json','',1).strip()
    return json.loads(c)

def chunk_all(docs):
    chunks = []
    for d in docs:
        toks = d['text'].split()
        for i in range(0, len(toks), CHUNK - OVERLAP):
            c = ' '.join(toks[i:i + CHUNK])
            if len(c.split()) > 20:
                chunks.append({'source': d['source'], 'text': c})
    return chunks

docs = json.load(open(CACHE, encoding='utf-8'))
chunks = chunk_all(docs)
texts = [c['text'] for c in chunks]
embedder = SentenceTransformer('all-MiniLM-L6-v2')
emb = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
qv = embedder.encode([QUERY], convert_to_numpy=True, normalize_embeddings=True)[0]
bm25 = BM25Okapi([t.lower().split() for t in texts])
sim = emb @ qv
bm = bm25.get_scores(QUERY.lower().split())

def retrieve_by_source(src, per):
    src_idx = [i for i in range(len(chunks)) if chunks[i]['source'] == src]
    src_dense = sorted(src_idx, key=lambda i: -sim[i])
    src_bm = sorted(src_idx, key=lambda i: -bm[i])
    score = {}
    for rank, i in enumerate(src_dense[:per * 2]): score[i] = score.get(i, 0) + 1.0 / (rank + 1)
    for rank, i in enumerate(src_bm[:per * 2]): score[i] = score.get(i, 0) + 1.0 / (rank + 1)
    return [texts[i] for i in sorted(score, key=lambda x: -score[x])[:per]]

context = retrieve_by_source('academic', 4) + retrieve_by_source('policy', 4)
print(f'检索 context: academic 4 + policy 4 = {len(context)} 段')

# 选 200 图
common = pd.read_csv(r'e:\PAPER1\human_sampled_matched_200.csv', encoding='utf-8-sig')
common['oid'] = common['Filename'].astype(str).str.split('_').str[0]
imgs = []
for oid in sorted(common['oid'].tolist()):
    m = glob.glob(os.path.join(IMG_DIR, f'{oid}_*.jpg'))
    if m: imgs.append((oid, m[0]))
print(f'选中 {len(imgs)} 张图')

# 生成（共同 + 分开，均 nano + full RAG，唯一差异=生成方式）
records = []
for idx, (oid, p) in enumerate(imgs):
    with open(p, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    # 共同生成：一次调用七维
    joint = None
    for attempt in range(3):
        try:
            joint = parse(call_llm(b64, build_prompt(DIMS_ALL, context)))['scores']
            break
        except Exception:
            if attempt == 2: raise
            time.sleep(3)
    # 分开生成：调用1 六维
    sc6 = None
    for attempt in range(3):
        try:
            sc6 = parse(call_llm(b64, build_prompt(DIMS6, context)))['scores']
            break
        except Exception:
            if attempt == 2: raise
            time.sleep(3)
    # 分开生成：调用2 walkability
    scw = None
    for attempt in range(3):
        try:
            scw = parse(call_llm(b64, build_prompt(['walkability'], context)))['scores']
            break
        except Exception:
            if attempt == 2: raise
            time.sleep(3)
    records.append({'oid': oid, 'joint': joint, 'six_dims': sc6, 'walkability': scw})
    print(f'[{idx+1}/{len(imgs)}] {oid}', flush=True)
    if (idx + 1) % 20 == 0:
        json.dump(records, open(OUT_RAW, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

json.dump(records, open(OUT_RAW, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'\n已保存 {OUT_RAW}')

rows = []
for r in records:
    row = {'oid': r['oid']}
    for d in DIMS6:
        row[f'joint_{d}'] = r['joint'].get(d)
        row[f'sep_{d}'] = r['six_dims'].get(d)
    row['joint_walkability'] = r['joint'].get('walkability')
    row['sep_walkability'] = r['walkability'].get('walkability')
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'separate_generation_200.csv'), index=False, encoding='utf-8-sig')
print('已保存 separate_generation_200.csv')
