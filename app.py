from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import routers
from routes import dp_routes
from routes import recommendation_routes

app = FastAPI(
    title="OrderEZ ML API",
    description="Machine Learning backend for OrderEZ - Dynamic Pricing & Recommendations",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dp_routes.router)
app.include_router(recommendation_routes.router)


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "OrderEZ - ML Backend API",
            "status": "running",
            "version": "2.0.0",
            "endpoints": {
                "dynamic_pricing": {
                    "train": {
                        "method": "POST",
                        "url": "/dp/train",
                        "description": "Train LSTM model for a branch",
                    },
                    "predict": {
                        "method": "POST",
                        "url": "/dp/predict",
                        "description": "Generate price predictions",
                    },
                    "predictions": {
                        "method": "GET",
                        "url": "/dp/predictions/{branch_id}",
                        "description": "Get stored predictions",
                    },
                    "model": {
                        "method": "GET",
                        "url": "/dp/model/{branch_id}",
                        "description": "Get model information",
                    },
                    "branches": {
                        "method": "GET",
                        "url": "/dp/branches",
                        "description": "Get all branches",
                    },
                    "apply_price": {
                        "method": "POST",
                        "url": "/dp/apply-price",
                        "description": "Apply predicted price",
                    },
                    "health": {
                        "method": "GET",
                        "url": "/dp/health",
                        "description": "Health check",
                    },
                },
                "recommendations": {
                    "cart_based": {
                        "method": "POST",
                        "url": "/recommendation/cart-based",
                        "description": "Get recommendations based on cart items",
                    },
                    "personalized": {
                        "method": "POST",
                        "url": "/recommendation/personalized",
                        "description": "Get personalized recommendations for user",
                    },
                    "user_history": {
                        "method": "GET",
                        "url": "/recommendation/user-history/{branch_id}",
                        "description": "Get user order history",
                    },
                    "popular": {
                        "method": "GET",
                        "url": "/recommendation/popular/{branch_id}",
                        "description": "Get popular items",
                    },
                    "health": {
                        "method": "GET",
                        "url": "/recommendation/health",
                        "description": "Health check",
                    },
                },
            },
        },
    )


@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "OrderEZ ML API",
            "version": "2.0.0",
            "features": ["dynamic_pricing", "recommendations"],
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True, log_level="info")
