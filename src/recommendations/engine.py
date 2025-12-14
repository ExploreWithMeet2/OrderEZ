from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationEngine:

    def __init__(self):
        self.min_support = 0.02  
        self.min_confidence = 0.3 
        
    async def get_cart_based_recommendations(
        self,
        cart_items: List[str],
        all_orders: List[Dict],
        n_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
 
        try:
            if not cart_items or not all_orders:
                logger.warning("Empty cart or no order history")
                return []
            
            item_pairs = defaultdict(int)
            item_counts = defaultdict(int)
            
            for order in all_orders:
                order_items = [item["itemId"] for item in order.get("items", [])]
                
                for item in order_items:
                    item_counts[item] += 1
                
                for i, item1 in enumerate(order_items):
                    for item2 in order_items[i+1:]:
                        pair = tuple(sorted([item1, item2]))
                        item_pairs[pair] += 1
            
            recommendations = defaultdict(float)
            total_orders = len(all_orders)
            
            for cart_item in cart_items:
                if cart_item not in item_counts:
                    continue
                    
                for (item1, item2), pair_count in item_pairs.items():
                    if cart_item not in (item1, item2):
                        continue
                    
                    other_item = item2 if item1 == cart_item else item1
                    
                    if other_item in cart_items:
                        continue
                    
                    confidence = pair_count / item_counts[cart_item]
                    
                    support = pair_count / total_orders
                    
                    lift = confidence / (item_counts[other_item] / total_orders)
                    
                    score = (confidence * 0.5) + (support * 0.3) + (min(lift, 3) / 3 * 0.2)
                    
                    recommendations[other_item] = max(recommendations[other_item], score)
            
            sorted_recs = sorted(
                recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            return [
                {
                    "itemId": item_id,
                    "score": round(score, 3),
                    "reason": "frequently_bought_together",
                    "confidence": round(score * 100, 1)
                }
                for item_id, score in sorted_recs
            ]
            
        except Exception as e:
            logger.error(f"Error in cart-based recommendations: {str(e)}")
            return []
    
    async def get_personalized_recommendations(
        self,
        user_orders: List[Dict],
        all_items: List[Dict],
        all_orders: List[Dict],
        n_recommendations: int = 5,
        include_new: bool = True
    ) -> List[Dict[str, Any]]:
        try:
            if not user_orders:
                return await self._get_popular_items(all_orders, all_items, n_recommendations)
            
            user_item_freq = defaultdict(int)
            user_item_recency = {}
            user_categories = defaultdict(int)
            
            for order in user_orders:
                order_time = order.get("updatedAt", 0)
                for item in order.get("items", []):
                    item_id = item["itemId"]
                    user_item_freq[item_id] += item.get("quantity", 1)
                    user_item_recency[item_id] = max(
                        user_item_recency.get(item_id, 0),
                        order_time
                    )
            
            item_similarity = self._calculate_item_similarity(all_orders)
            
            recommendations = {}
            current_time = datetime.now().timestamp() * 1000
            
            for item in all_items:
                item_id = item["_id"]
                
                if not item.get("isAvailable", True):
                    continue
                
                if item_id in user_item_recency:
                    days_since = (current_time - user_item_recency[item_id]) / (1000 * 60 * 60 * 24)
                    if days_since < 7 and user_item_freq[item_id] < 3:
                        continue
                
                score = 0.0
                
                collab_score = 0.0
                for ordered_item, freq in user_item_freq.items():
                    similarity = item_similarity.get((ordered_item, item_id), 0)
                    collab_score += similarity * freq
                
                if user_item_freq:
                    collab_score /= sum(user_item_freq.values())
                
                popularity = self._get_item_popularity(item_id, all_orders)
                
                trending_score = self._get_trending_score(item_id, all_orders)
                
                content_score = 0.0
                item_tags = item.get("tags", [])
                for tag in item_tags:
                    if tag in user_categories:
                        content_score += 0.2
                
                score = (
                    collab_score * 0.4 +
                    popularity * 0.25 +
                    trending_score * 0.2 +
                    content_score * 0.15
                )
                
                if include_new and item_id not in user_item_freq:
                    score *= 1.1
                
                avg_user_price = self._get_avg_user_price(user_orders)
                item_price = item.get("currentPrice", item.get("basePrice", 0))
                if item_price > avg_user_price * 1.5 and item_id not in user_item_freq:
                    score *= 0.8
                
                recommendations[item_id] = score
            
            sorted_recs = sorted(
                recommendations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            return [
                {
                    "itemId": item_id,
                    "score": round(score, 3),
                    "reason": self._get_recommendation_reason(item_id, user_item_freq, all_orders),
                    "confidence": round(min(score * 100, 100), 1),
                    "isNew": item_id not in user_item_freq
                }
                for item_id, score in sorted_recs if score > 0
            ]
            
        except Exception as e:
            logger.error(f"Error in personalized recommendations: {str(e)}")
            return []
    
    def _calculate_item_similarity(self, all_orders: List[Dict]) -> Dict[Tuple[str, str], float]:
        item_pairs = defaultdict(int)
        item_counts = defaultdict(int)
        
        for order in all_orders:
            order_items = [item["itemId"] for item in order.get("items", [])]
            
            for item in order_items:
                item_counts[item] += 1
            
            for i, item1 in enumerate(order_items):
                for item2 in order_items[i+1:]:
                    if item1 != item2:
                        pair = tuple(sorted([item1, item2]))
                        item_pairs[pair] += 1
        
        similarities = {}
        for (item1, item2), co_occur in item_pairs.items():
            denominator = item_counts[item1] + item_counts[item2] - co_occur
            if denominator > 0:
                similarity = co_occur / denominator
                similarities[(item1, item2)] = similarity
                similarities[(item2, item1)] = similarity
        
        return similarities
    
    def _get_item_popularity(self, item_id: str, all_orders: List[Dict]) -> float:
        item_count = 0
        total_items = 0
        
        for order in all_orders:
            for item in order.get("items", []):
                total_items += 1
                if item["itemId"] == item_id:
                    item_count += item.get("quantity", 1)
        
        return item_count / max(total_items, 1)
    
    def _get_trending_score(self, item_id: str, all_orders: List[Dict]) -> float:
        try:
            current_time = datetime.now().timestamp() * 1000
            recent_cutoff = current_time - (7 * 24 * 60 * 60 * 1000)  
            
            recent_count = 0
            total_recent = 0
            
            for order in all_orders:
                order_time = order.get("updatedAt", 0)
                if order_time >= recent_cutoff:
                    total_recent += 1
                    for item in order.get("items", []):
                        if item["itemId"] == item_id:
                            recent_count += 1
            
            return recent_count / max(total_recent, 1)
        except Exception:
            return 0.0
    
    def _get_avg_user_price(self, user_orders: List[Dict]) -> float:
        if not user_orders:
            return 100.0  
        
        total_price = 0
        total_items = 0
        
        for order in user_orders:
            for item in order.get("items", []):
                total_price += item.get("price", 0) * item.get("quantity", 1)
                total_items += item.get("quantity", 1)
        
        return total_price / max(total_items, 1)
    
    def _get_recommendation_reason(
        self,
        item_id: str,
        user_item_freq: Dict[str, int],
        all_orders: List[Dict]
    ) -> str:
        if item_id not in user_item_freq:
            return "popular_choice"
        elif user_item_freq[item_id] >= 3:
            return "your_favorite"
        else:
            return "based_on_preferences"
    
    async def _get_popular_items(
        self,
        all_orders: List[Dict],
        all_items: List[Dict],
        n: int
    ) -> List[Dict[str, Any]]:
        try:
            item_popularity = defaultdict(int)
            
            for order in all_orders:
                for item in order.get("items", []):
                    item_popularity[item["itemId"]] += item.get("quantity", 1)
            
            # Filter to available items
            available_items = {item["_id"] for item in all_items if item.get("isAvailable", True)}
            
            sorted_items = sorted(
                [(k, v) for k, v in item_popularity.items() if k in available_items],
                key=lambda x: x[1],
                reverse=True
            )[:n]
            
            return [
                {
                    "itemId": item_id,
                    "score": 0.5,
                    "reason": "most_popular",
                    "confidence": 50.0,
                    "isNew": True
                }
                for item_id, _ in sorted_items
            ]
        except Exception as e:
            logger.error(f"Error getting popular items: {str(e)}")
            return []


recommendation_engine = RecommendationEngine()