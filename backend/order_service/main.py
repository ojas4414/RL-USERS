import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

CART_SERVICE = "http://cart_service:8002"
PRODUCT_SERVICE = "http://product_service:8001"


class OrderRequest(BaseModel):
    agent_id: int
    product_id: str
    balance: float


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/order")
async def place_order(req: OrderRequest):
    async with httpx.AsyncClient() as client:
        cart_resp = await client.get(f"{CART_SERVICE}/cart/{req.agent_id}")
        cart = cart_resp.json()
        if not cart:
            raise HTTPException(status_code=400, detail="Cart is empty")

        prod_resp = await client.get(f"{PRODUCT_SERVICE}/products/{req.product_id}")
        if prod_resp.status_code == 404:
            raise HTTPException(status_code=400, detail="Product not found")
        product = prod_resp.json()

        if product["stock"] <= 0:
            raise HTTPException(status_code=400, detail="Product out of stock")

        price = product["price"]
        if req.balance < price:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        dec_resp = await client.post(
            f"{PRODUCT_SERVICE}/products/{req.product_id}/decrement_stock"
        )
        if dec_resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to decrement stock")

        await client.delete(f"{CART_SERVICE}/cart/{req.agent_id}")

    return {"success": True, "new_balance": round(req.balance - price, 2)}
