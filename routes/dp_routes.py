from pathlib import Path
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from src.dp.train import train
from src.dp.predict import predict_prices
from src.dp.convex_caller.main import (
    get_all_branches,
    get_latest_predictions,
    get_active_model,
    update_item_price,
)

router = APIRouter(prefix="/dp", tags=["Dynamic Pricing"])


class TrainRequest(BaseModel):
    branch_id: str


class PredictRequest(BaseModel):
    branch_id: str


class ApplyPriceRequest(BaseModel):
    item_id: str
    new_price: float
    prediction_id: str = None


@router.post("/train")
async def train_model_route():

    try:
        result = await train("train")

        if result["type"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": result["message"],
                "data": result["data"],
            },
        )

    except Exception as e:
        raise HTTPException(status_code=200, detail=str(e))


@router.post("/predict")
async def predict_prices_route(request: PredictRequest):
    try:
        result = await predict_prices(request.branch_id)

        if result["type"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": result["message"],
                "data": result["data"],
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/branches")
async def get_branches():
    try:
        result = await get_all_branches()

        if result["type"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return JSONResponse(
            status_code=200, content={"status": "success", "branches": result["data"]}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{branch_id}")
async def get_predictions(branch_id: str, limit: int = Query(default=20, ge=1, le=100)):
    try:
        result = await get_latest_predictions(branch_id, limit)

        if result["type"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return JSONResponse(
            status_code=200,
            content={"status": "success", "predictions": result["data"]},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/{branch_id}")
async def get_model_info(branch_id: str):
    try:
        result = await get_active_model(branch_id)

        if result["type"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        model_data = result["data"]

        if not model_data:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "not_found",
                    "message": "No active model found for this branch",
                },
            )

        return JSONResponse(
            status_code=200, content={"status": "success", "model": model_data}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-price")
async def apply_price_change(request: ApplyPriceRequest):
    try:
        result = await update_item_price(
            request.item_id, request.new_price, request.prediction_id
        )

        if result["type"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        return JSONResponse(
            status_code=200, content={"status": "success", "message": result["message"]}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    models_dir = Path("models")

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "models_directory": str(models_dir),
            "models_exist": models_dir.exists(),
        },
    )
