import hashlib
import json
import random
from collections import Counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

DATA_FILE = "data/raw/All_Beauty.jsonl"


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    stock: int


def _deterministic_price(asin: str) -> float:
    # md5 gives a stable seed across Python runs (unlike hash())
    seed = int(hashlib.md5(asin.encode()).hexdigest(), 16) % (2 ** 32)
    return round(random.Random(seed).uniform(10, 500), 2)


def _load_catalog(filepath: str, top_n: int = 500) -> list[Product]:
    counter: Counter = Counter()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                asin = json.loads(line).get("parent_asin")
                if asin:
                    counter[asin] += 1
    except FileNotFoundError:
        return []

    return [
        Product(
            product_id=asin,
            name=asin,
            category="beauty",
            price=_deterministic_price(asin),
            stock=100,
        )
        for asin, _ in counter.most_common(top_n)
    ]


_catalog: list[Product] = _load_catalog(DATA_FILE)
_index: dict[str, Product] = {p.product_id: p for p in _catalog}


@app.get("/health")
def health():
    return {"status": "alive", "catalog_size": len(_catalog)}


@app.get("/products", response_model=list[Product])
def get_products():
    return _catalog


@app.get("/products/category/{category}", response_model=list[Product])
def get_by_category(category: str):
    return [p for p in _catalog if p.category == category]


@app.get("/products/{product_id}", response_model=Product)
def get_product(product_id: str):
    p = _index.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    return p


@app.post("/products/{product_id}/decrement_stock")
def decrement_stock(product_id: str):
    p = _index.get(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Product not found")
    if p.stock <= 0:
        raise HTTPException(status_code=400, detail="Out of stock")
    p.stock -= 1
    return {"product_id": product_id, "stock": p.stock}
