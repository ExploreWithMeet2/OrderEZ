from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class RecommendationCache:
    """Simple in-memory cache for recommendations"""
    
    def __init__(self, ttl_seconds: int = 300):  
        self.cache = {}
        self.ttl_seconds = ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.ttl_seconds:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value with timestamp"""
        self.cache[key] = (value, datetime.now())
    
    def clear(self):
        """Clear all cached values"""
        self.cache.clear()


recommendation_cache = RecommendationCache()


def calculate_time_decay(timestamp: float, decay_rate: float = 0.1) -> float:
    try:
        current_time = datetime.now().timestamp() * 1000
        days_ago = (current_time - timestamp) / (1000 * 60 * 60 * 24)
        
        decay = max(0, 1 - (decay_rate * days_ago))
        return decay
    except Exception:
        return 0.5  


def filter_available_items(items: List[Dict]) -> List[Dict]:
    """Filter to only available items"""
    return [item for item in items if item.get("isAvailable", True)]


def calculate_diversity_score(recommendations: List[Dict], items: List[Dict]) -> float:
    try:
        if not recommendations:
            return 0.0
        
        item_lookup = {item["_id"]: item for item in items}
        
        categories = set()
        for rec in recommendations:
            item_id = rec.get("itemId")
            if item_id in item_lookup:
                item_tags = item_lookup[item_id].get("tags", [])
                categories.update(item_tags)
        
        diversity = len(categories) / len(recommendations)
        return min(diversity, 1.0)
        
    except Exception as e:
        logger.error(f"Error calculating diversity: {str(e)}")
        return 0.0


def get_price_tier(price: float) -> str:
    """Categorize item into price tier"""
    if price < 50:
        return "budget"
    elif price < 150:
        return "mid"
    elif price < 250:
        return "premium"
    else:
        return "luxury"


def format_recommendation_reason(reason: str) -> str:
    """Convert reason code to human-readable message"""
    reason_map = {
        "frequently_bought_together": "Frequently bought together with your cart items",
        "your_favorite": "One of your favorites",
        "based_on_preferences": "Based on your preferences",
        "popular_choice": "Popular among customers",
        "trending": "Trending now",
        "similar_to_ordered": "Similar to items you've ordered",
        "most_popular": "Most popular item"
    }
    return reason_map.get(reason, "Recommended for you")


def batch_items(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
    """Split items into batches for processing"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def calculate_confidence_interval(score: float, sample_size: int) -> Dict[str, float]:
    import math
    
    if sample_size == 0:
        return {"lower": 0, "upper": 0, "confidence": 0}
    
    z = 1.96  
    
    phat = score
    denominator = 1 + z**2 / sample_size
    centre = (phat + z**2 / (2 * sample_size)) / denominator
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * sample_size)) / sample_size) / denominator
    
    return {
        "lower": max(0, centre - margin),
        "upper": min(1, centre + margin),
        "confidence": score
    }


def get_recommendation_explanation(
    rec: Dict,
    user_orders: List[Dict],
    all_orders: List[Dict]
) -> str:
    try:
        reason = rec.get("reason", "")
        score = rec.get("score", 0)
        item_id = rec.get("itemId")
        
        explanations = []
        
        user_item_count = sum(
            1 for order in user_orders
            for item in order.get("items", [])
            if item["itemId"] == item_id
        )
        
        if user_item_count > 0:
            explanations.append(f"You've ordered this {user_item_count} times")
        
        total_orders = sum(
            1 for order in all_orders
            for item in order.get("items", [])
            if item["itemId"] == item_id
        )
        
        if total_orders > len(all_orders) * 0.1:
            explanations.append("Very popular among customers")
        
        confidence = rec.get("confidence", 0)
        if confidence > 80:
            explanations.append("High confidence match")
        elif confidence > 60:
            explanations.append("Good match")
        
        if explanations:
            return " • ".join(explanations)
        else:
            return format_recommendation_reason(reason)
            
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return "Recommended for you"


def validate_recommendation_request(
    branch_id: str,
    n_recommendations: int,
    max_recommendations: int = 20
) -> tuple[bool, Optional[str]]:
    if not branch_id or not isinstance(branch_id, str):
        return False, "Invalid branch_id"
    
    if n_recommendations < 1:
        return False, "n_recommendations must be at least 1"
    
    if n_recommendations > max_recommendations:
        return False, f"n_recommendations cannot exceed {max_recommendations}"
    
    return True, None


def merge_recommendations(
    cart_recs: List[Dict],
    personalized_recs: List[Dict],
    weight_cart: float = 0.6
) -> List[Dict]:
    try:
        scores = {}
        
        for rec in cart_recs:
            item_id = rec.get("itemId")
            if item_id:
                scores[item_id] = {
                    **rec,
                    "score": rec.get("score", 0) * weight_cart
                }
        
        weight_personalized = 1 - weight_cart
        for rec in personalized_recs:
            item_id = rec.get("itemId")
            if item_id:
                if item_id in scores:
                    # Combine scores
                    scores[item_id]["score"] += rec.get("score", 0) * weight_personalized
                    scores[item_id]["reason"] = "cart_and_preferences"
                else:
                    scores[item_id] = {
                        **rec,
                        "score": rec.get("score", 0) * weight_personalized
                    }
        
        merged = sorted(
            scores.values(),
            key=lambda x: x.get("score", 0),
            reverse=True
        )
        
        return merged
        
    except Exception as e:
        logger.error(f"Error merging recommendations: {str(e)}")
        return cart_recs + personalized_recs


__all__ = [
    "RecommendationCache",
    "recommendation_cache",
    "calculate_time_decay",
    "filter_available_items",
    "calculate_diversity_score",
    "get_price_tier",
    "format_recommendation_reason",
    "batch_items",
    "calculate_confidence_interval",
    "get_recommendation_explanation",
    "validate_recommendation_request",
    "merge_recommendations"
]