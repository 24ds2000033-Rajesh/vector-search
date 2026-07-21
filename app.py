from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import json

app = FastAPI(title="Two Stage Vector Search")

# --------------------------
# Load datasets once
# --------------------------

documents = pd.read_csv("documents.csv")

with open("embeddings.json", "r") as f:
    embeddings = json.load(f)

with open("reranker_scores.json", "r") as f:
    reranker_scores = json.load(f)


# --------------------------
# Request Model
# --------------------------

class SearchRequest(BaseModel):
    query_id: str
    query_vector: List[float]
    top_k: int
    rerank_top_n: int
    filter: Dict[str, Any]


# --------------------------
# Metadata Filtering
# --------------------------

def apply_filters(df, filters):

    result = df.copy()

    for field, condition in filters.items():

        if isinstance(condition, dict):

            if "gte" in condition:
                result = result[result[field] >= condition["gte"]]

            if "lte" in condition:
                result = result[result[field] <= condition["lte"]]

            if "in" in condition:
                result = result[result[field].isin(condition["in"])]

        else:
            result = result[result[field] == condition]

    return result


# --------------------------
# Cosine Similarity
# --------------------------

def cosine_similarity(v1, v2):

    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


# --------------------------
# Endpoint
# --------------------------

@app.post("/vector-search")
def vector_search(req: SearchRequest):

    if req.query_id not in reranker_scores:
        raise HTTPException(404, "Unknown query_id")

    filtered_docs = apply_filters(documents, req.filter)

    similarities = []

    for _, row in filtered_docs.iterrows():

        doc_id = row["doc_id"]

        if doc_id not in embeddings:
            continue

        score = cosine_similarity(
            req.query_vector,
            embeddings[doc_id]
        )

        similarities.append((doc_id, score))

    # Stage 1 ranking
    similarities.sort(
        key=lambda x: (-x[1], x[0])
    )

    stage1 = similarities[:req.top_k]

    rerank_table = reranker_scores[req.query_id]

    reranked = []

    for doc_id, _ in stage1:

        score = rerank_table.get(doc_id, float("-inf"))

        reranked.append((doc_id, score))

    # Stage 2 ranking
    reranked.sort(
        key=lambda x: (-x[1], x[0])
    )

    answer = [doc for doc, _ in reranked[:req.rerank_top_n]]

    return {
        "matches": answer
    }


@app.get("/")
def root():
    return {"status": "ok"}
