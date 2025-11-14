from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from schema.convex_schema import ConvexRequest
from utils.convex import call_convex
from src.recommendations.engine import RecommendationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recommendation", tags=["Recommendations"])

class CartRecommendationRequest(BaseModel):
    """Request model for cart-based recommendations"""
    branch_id: str = Field(..., description="Branch ID")
    cart_items: List[str] = Field(..., description="List of item IDs in cart")
    n_recommendations: int = Field(default=5, ge=1, le=20, description="Number of recommendations")


class PersonalizedRecommendationRequest(BaseModel):
    """Request model for personalized recommendations"""
    branch_id: str = Field(..., description="Branch ID")
    user_name: str = Field(..., description="Username for personalization")
    n_recommendations: int = Field(default=5, ge=1, le=20, description="Number of recommendations")
    include_new: bool = Field(default=True, description="Include items user hasn't tried")


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    success: bool
    message: str
    recommendations: List[dict]
    metadata: Optional[dict] = None


@router.post("/cart-based")
async def get_cart_recommendations(request: CartRecommendationRequest):
    try:
        logger.info(f"Cart recommendations request for branch: {request.branch_id}")
        
        if not request.cart_items:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "Cart is empty",
                    "recommendations": [],
                    "metadata": {"cart_size": 0}
                }
            )
        
        orders_response = await call_convex(
            ConvexRequest(
                module="orders",
                func="getByBranch",
                isQuery=True,
                args={"branchId": request.branch_id},
                returnDf=False,
            )
        )
        
        if orders_response.get("type") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch orders: {orders_response.get('error')}"
            )
        
        all_orders = orders_response.get("data", [])
        
        if not all_orders:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "No order history available",
                    "recommendations": [],
                    "metadata": {"total_orders": 0}
                }
            )
        
        recommendations = await RecommendationEngine.get_cart_based_recommendations(
            cart_items=request.cart_items,
            all_orders=all_orders,
            n_recommendations=request.n_recommendations
        )
        
        enriched_recommendations = await _enrich_recommendations(
            recommendations,
            request.branch_id
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Found {len(enriched_recommendations)} recommendations",
                "recommendations": enriched_recommendations,
                "metadata": {
                    "cart_size": len(request.cart_items),
                    "total_orders": len(all_orders),
                    "algorithm": "association_rules"
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error generating cart recommendations: {str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e),
                "recommendations": []
            }
        )


@router.post("/personalized")
async def get_personalized_recommendations(request: PersonalizedRecommendationRequest):
    try:
        logger.info(f"Personalized recommendations for user: {request.user_name}")
        
        user_orders_response = await call_convex(
            ConvexRequest(
                module="orders",
                func="getOrdersByBranchAndUser",
                isQuery=True,
                args={
                    "branchId": request.branch_id,
                    "userName": request.user_name
                },
                returnDf=False,
            )
        )
        
        if user_orders_response.get("type") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch user orders: {user_orders_response.get('error')}"
            )
        
        user_orders = user_orders_response.get("data", [])
        
        all_orders_response = await call_convex(
            ConvexRequest(
                module="orders",
                func="getByBranch",
                isQuery=True,
                args={"branchId": request.branch_id},
                returnDf=False,
            )
        )
        
        if all_orders_response.get("type") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch all orders: {all_orders_response.get('error')}"
            )
        
        all_orders = all_orders_response.get("data", [])
        
        all_items_response = await call_convex(
            ConvexRequest(
                module="items",
                func="getByBranch",
                isQuery=True,
                args={"branchId": request.branch_id},
                returnDf=False,
            )
        )
        
        if all_items_response.get("type") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch items: {all_items_response.get('error')}"
            )
        
        all_items = all_items_response.get("data", [])
        
        if not all_items:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": "No items available",
                    "recommendations": [],
                    "metadata": {"total_items": 0}
                }
            )
        
        recommendations = await RecommendationEngine.get_personalized_recommendations(
            user_name=request.user_name,
            user_orders=user_orders,
            all_items=all_items,
            all_orders=all_orders,
            n_recommendations=request.n_recommendations,
            include_new=request.include_new
        )
        
        enriched_recommendations = await _enrich_recommendations(
            recommendations,
            request.branch_id
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Found {len(enriched_recommendations)} personalized recommendations",
                "recommendations": enriched_recommendations,
                "metadata": {
                    "user_name": request.user_name,
                    "user_order_count": len(user_orders),
                    "total_orders": len(all_orders),
                    "total_items": len(all_items),
                    "algorithm": "collaborative_content_hybrid",
                    "is_new_user": len(user_orders) == 0
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error generating personalized recommendations: {str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e),
                "recommendations": []
            }
        )


@router.get("/user-history/{branch_id}")
async def get_user_history(
    branch_id: str,
    user_name: str = Query(..., description="Username"),
    limit: int = Query(default=10, ge=1, le=50, description="Max orders to return")
):
    try:
        logger.info(f"Fetching history for user: {user_name}")
        
        response = await call_convex(
            ConvexRequest(
                module="orders",
                func="getOrdersByBranchAndUser",
                isQuery=True,
                args={
                    "branchId": branch_id,
                    "userName": user_name
                },
                returnDf=False,
            )
        )
        
        if response.get("type") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch orders: {response.get('error')}"
            )
        
        orders = response.get("data", [])[:limit]
        
        total_spent = sum(order.get("total", 0) for order in orders)
        total_items = sum(len(order.get("items", [])) for order in orders)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Found {len(orders)} orders",
                "orders": orders,
                "statistics": {
                    "total_orders": len(orders),
                    "total_spent": round(total_spent, 2),
                    "total_items": total_items,
                    "avg_order_value": round(total_spent / max(len(orders), 1), 2)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error fetching user history: {str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e)
            }
        )


@router.get("/popular/{branch_id}")
async def get_popular_items(
    branch_id: str,
    limit: int = Query(default=10, ge=1, le=50, description="Number of items")
):
    try:
        logger.info(f"Fetching popular items for branch: {branch_id}")
        
        orders_response = await call_convex(
            ConvexRequest(
                module="orders",
                func="getByBranch",
                isQuery=True,
                args={"branchId": branch_id},
                returnDf=False,
            )
        )
        
        if orders_response.get("type") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch orders: {orders_response.get('error')}"
            )
        
        all_orders = orders_response.get("data", [])
        
        from collections import defaultdict
        item_stats = defaultdict(lambda: {"count": 0, "revenue": 0})
        
        for order in all_orders:
            for item in order.get("items", []):
                item_id = item["itemId"]
                quantity = item.get("quantity", 1)
                price = item.get("price", 0)
                
                item_stats[item_id]["count"] += quantity
                item_stats[item_id]["revenue"] += price * quantity
        
        sorted_items = sorted(
            item_stats.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:limit]
        
        popular_items = []
        for item_id, stats in sorted_items:
            item_response = await call_convex(
                ConvexRequest(
                    module="items",
                    func="getById",
                    isQuery=True,
                    args={"itemId": item_id},
                    returnDf=False,
                )
            )
            
            if item_response.get("type") != "error" and item_response.get("data"):
                item_data = item_response["data"]
                popular_items.append({
                    **item_data,
                    "popularity_stats": {
                        "total_orders": stats["count"],
                        "total_revenue": round(stats["revenue"], 2)
                    }
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Found {len(popular_items)} popular items",
                "items": popular_items
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error fetching popular items: {str(e)}"
        logger.error(error_msg)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e)
            }
        )


async def _enrich_recommendations(
    recommendations: List[dict],
    branch_id: str
) -> List[dict]:
    enriched = []
    
    for rec in recommendations:
        try:
            item_id = rec.get("itemId")
            if not item_id:
                continue
            
            item_response = await call_convex(
                ConvexRequest(
                    module="items",
                    func="getById",
                    isQuery=True,
                    args={"itemId": item_id},
                    returnDf=False,
                )
            )
            
            if item_response.get("type") == "error" or not item_response.get("data"):
                logger.warning(f"Could not fetch item {item_id}")
                continue
            
            item_data = item_response["data"]
            
            if not item_data.get("isAvailable", True):
                continue
            
            enriched.append({
                "item": item_data,
                "recommendation_score": rec.get("score", 0),
                "confidence": rec.get("confidence", 0),
                "reason": rec.get("reason", "recommended"),
                "is_new": rec.get("isNew", False)
            })
            
        except Exception as e:
            logger.error(f"Error enriching recommendation: {str(e)}")
            continue
    
    return enriched