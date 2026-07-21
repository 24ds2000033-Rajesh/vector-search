from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import json

app = FastAPI()

# Load once
documents = pd.read_csv("documents.csv")

with open("embeddings.json") as f:
    embeddings = json.load(f)

with open("reranker_scores.json") as f:
    reranker_scores = json.load(f)


class SearchRequest(BaseModel):
    query_id: str
    query_vector: List[float]
    top_k: int
    rerank_top_n: int
    filter: Dict[str, Any]


def apply_filter(df, filters):

    result = df

    for key, value in filters.items():

        if isinstance(value, dict):

            if "gte" in value:
                result = result[result[key] >= value["gte"]]

            if "lte" in value:
                result = result[result[key] <= value["lte"]]

            if "in" in value:
                result = result[result[key].isin(value["in"])]

        else:
            result = result[result[key] == value]

    return result


def cosine(q, d):

    q = np.asarray(q, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    denom = np.linalg.norm(q) * np.linalg.norm(d)

    if denom == 0:
        return 0.0

    return float(np.dot(q, d) / denom)


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/vector-search")
def vector_search(req: SearchRequest):

    if req.query_id not in reranker_scores:
        raise HTTPException(404, "Unknown query")

    filtered = apply_filter(documents, req.filter)

    stage1 = []

    for _, row in filtered.iterrows():

        doc_id = row.doc_id

        emb = embeddings.get(doc_id)

        if emb is None:
            continue

        sim = cosine(req.query_vector, emb)

        stage1.append(
            {
                "doc_id": doc_id,
                "score": sim
            }
        )

    # Stage 1
    stage1.sort(
        key=lambda x: (-x["score"], x["doc_id"])
    )

    stage1 = stage1[:req.top_k]

    scores = reranker_scores[req.query_id]

    stage2 = []

    for item in stage1:

        doc = item["doc_id"]

        stage2.append(
            {
                "doc_id": doc,
                "score": scores.get(doc, float("-inf"))
            }
        )

    # Stage 2
    stage2.sort(
        key=lambda x: (-x["score"], x["doc_id"])
    )

    return {
        "matches": [
            x["doc_id"]
            for x in stage2[:req.rerank_top_n]
        ]
    }
