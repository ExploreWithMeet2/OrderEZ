from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from keras.models import load_model

from src.dp.convex_caller.main import (
    fetch_branch_items,
    fetch_price_history,
    fetch_item_metrics,
    store_predictions_batch,
    create_owner_alert
)
from src.dp.preprocessing import (
    load_preprocessor,
    transform,
    calculate_price_change_percent,
    calculate_demand_metrics,
)
from utils.returnFormat import returnFormat

SEQUENCE_LENGTH = 30
MAX_PRICE_CHANGE = 0.20  
MIN_CONFIDENCE = 0.6


def categorize_demand(predicted_change: float) -> str:
    if predicted_change > 10:
        return "HIGH"
    elif predicted_change < -5:
        return "LOW"
    else:
        return "NORMAL"


def generate_recommendation(
    current_price: float,
    predicted_change_percent: float,
    confidence: float,
    demand_category: str
) -> Dict:
    price_change = current_price * (predicted_change_percent / 100)
    
    if demand_category == "HIGH" and predicted_change_percent > 5:
        action = "INCREASE"
        reason = f"High demand detected. Market willing to pay {predicted_change_percent:.1f}% more."
        alert_level = "success"
        should_apply = confidence > 0.7
    
    elif demand_category == "LOW" and predicted_change_percent < -3:
        action = "ALERT"
        reason = f"Low demand detected. Price may need to decrease by {abs(predicted_change_percent):.1f}%."
        alert_level = "danger"
        should_apply = False
    
    elif abs(predicted_change_percent) < 3:
        action = "MAINTAIN"
        reason = "Current pricing is optimal for current demand."
        alert_level = "info"
        should_apply = False
    
    elif predicted_change_percent > 3:
        action = "INCREASE"
        reason = f"Moderate demand increase. Consider raising price by {predicted_change_percent:.1f}%."
        alert_level = "warning"
        should_apply = confidence > 0.65
    
    else:
        action = "DECREASE"
        reason = f"Demand softening. Consider lowering price by {abs(predicted_change_percent):.1f}%."
        alert_level = "warning"
        should_apply = confidence > 0.65
    
    return {
        "action": action,
        "reason": reason,
        "priceChange": float(price_change),
        "percentChange": float(predicted_change_percent),
        "shouldApply": should_apply,
        "alertLevel": alert_level
    }


async def prepare_prediction_data(branch_id: str) -> pd.DataFrame:
    print(f"\nPreparing prediction data for branch {branch_id}...")
    
    items_response = await fetch_branch_items(branch_id)
    if items_response["type"] == "error":
        raise ValueError(items_response["message"])
    
    items_list = items_response["data"]
    if not items_list:
        raise ValueError(f"No items found for branch {branch_id}")
    
    print(f"  Found {len(items_list)} items")
    
    all_data = []
    
    for item in items_list:
        item_id = item["_id"]
        
        prices_response = await fetch_price_history(item_id, days=30)
        if prices_response["type"] == "error":
            continue
        
        prices_list = prices_response["data"]
        if not prices_list or len(prices_list) < SEQUENCE_LENGTH:
            print(f"  Skipping {item['name']}: insufficient history ({len(prices_list)} records)")
            continue
        
        metrics_response = await fetch_item_metrics(item_id, 7)
        metrics = (
            metrics_response["data"] if metrics_response["type"] == "success" else {}
        )
        
        for price_row in prices_list:
            timestamp = price_row["updatedAt"]
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            record = {
                "item_id": item_id,
                "branch_id": branch_id,
                "current_price": price_row["value"],
                "base_price": item.get("basePrice", price_row["value"]),
                "timestamp": timestamp,
                "dt": dt,
                "demand_7d": price_row.get("demand", "MEDIUM"),
                "rating_7d": 4,
                "orders_7d": metrics.get("totalOrders", 0),
                "revenue_7d": metrics.get("totalRevenue", 0),
                "avg_quantity": metrics.get("totalQuantity", 0) / max(metrics.get("totalOrders", 1), 1),
                "time_of_day": (
                    "NOON" if 12 <= dt.hour < 15 else
                    "MORNING" if 5 <= dt.hour < 12 else
                    "AFTERNOON" if 15 <= dt.hour < 20 else "NIGHT"
                ),
                "season": (
                    "WINTER" if dt.month in [12, 1, 2] else
                    "SUMMER" if dt.month in [3, 4, 5, 6] else "MONSOON"
                ),
                "day_of_week": dt.weekday(),
                "is_weekend": 1 if dt.weekday() >= 5 else 0,
                "is_holiday": price_row.get("isEvent", False),
                "is_event": price_row.get("isEvent", False),
                "event_name": "",
            }
            all_data.append(record)
    
    df = pd.DataFrame(all_data)
    print(f"  Prepared {len(df)} records for {df['item_id'].nunique()} items")
    
    return df


async def predict_prices(branch_id: str) -> dict:
    try:
        print("\n" + "="*70)
        print(f"GENERATING PREDICTIONS FOR BRANCH: {branch_id}")
        print("="*70)
        
        model_path = Path("models") / branch_id / "model.h5"
        preprocessor_path = Path("models") / branch_id / "preprocessor.pkl"
        
        if not model_path.exists():
            return returnFormat('error', f"No model found for branch {branch_id}. Train a model first.")
        
        if not preprocessor_path.exists():
            return returnFormat('error', f"No preprocessor found for branch {branch_id}.")
        
        model = load_model(model_path)
        
        load_preprocessor(preprocessor_path)
        df = await prepare_prediction_data(branch_id)
        
        df = calculate_price_change_percent(df)
        df = calculate_demand_metrics(df)
        transformed = transform(df)
        
        
        predictions = []
        
        exclude_cols = [
            'price_change_percent', 'item_id', 'timestamp', 'dt', 
            'event_name', 'branch_id', 'base_price'
        ]
        feature_cols = [col for col in transformed.columns if col not in exclude_cols]
        
        for item_id in transformed['item_id'].unique():
            item_data = transformed[transformed['item_id'] == item_id].sort_values('timestamp')
            
            if len(item_data) < SEQUENCE_LENGTH:
                continue
            
            last_sequence = item_data[feature_cols].values[-SEQUENCE_LENGTH:]
            X = np.array([last_sequence])
            
            predicted_change = model.predict(X, verbose=0)[0][0]
            
            predicted_change = np.clip(predicted_change, -MAX_PRICE_CHANGE * 100, MAX_PRICE_CHANGE * 100)
            
            confidence = 1.0 - (abs(predicted_change) / 50.0)  # 50% is max expected
            confidence = np.clip(confidence, 0.5, 1.0)
            
            current_price = item_data.iloc[-1]['current_price']
            
            demand_category = categorize_demand(predicted_change)
            
            suggested_price = current_price * (1 + predicted_change / 100)
            
            recommendation = generate_recommendation(
                current_price,
                predicted_change,
                confidence,
                demand_category
            )
            
            prediction = {
                "item_id": item_id,
                "branch_id": branch_id,
                "predicted_at": datetime.now().timestamp() * 1000,
                "predicted_change_percent": float(predicted_change),
                "confidence": float(confidence),
                "current_price": float(current_price),
                "suggested_price": float(suggested_price),
                "demand_category": demand_category,
                "recommendation": recommendation
            }
            
            predictions.append(prediction)
            
            if demand_category == "LOW":
                await create_owner_alert(
                    branch_id=branch_id,
                    item_id=item_id,
                    alert_type="LOW_DEMAND",
                    severity="warning",
                    message=f"Low demand detected. Predicted price change: {predicted_change:.1f}%",
                    predicted_change_percent=float(predicted_change),
                    confidence=float(confidence)
                )
        
        store_result = await store_predictions_batch(predictions)
        
        if store_result['type'] == 'error':
            print(f"Warning: Failed to store predictions: {store_result['message']}")
        else:
            print(f"✓ Stored {store_result['data']['successful']} predictions")
        
        # Summarize
        high_demand = len([p for p in predictions if p['demand_category'] == 'HIGH'])
        low_demand = len([p for p in predictions if p['demand_category'] == 'LOW'])
        normal_demand = len([p for p in predictions if p['demand_category'] == 'NORMAL'])
        
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        print(f"Total Items:     {len(predictions)}")
        print(f"High Demand:     {high_demand} items")
        print(f"Normal Demand:   {normal_demand} items")
        print(f"Low Demand:      {low_demand} items (⚠️  alerts created)")
        print("="*70)
        
        return returnFormat(
            'success',
            f"Generated {len(predictions)} predictions",
            {
                'predictions': predictions,
                'summary': {
                    'total': len(predictions),
                    'high_demand': high_demand,
                    'normal_demand': normal_demand,
                    'low_demand': low_demand
                }
            }
        )
    
    except Exception as e:
        import traceback
        error_msg = f"Prediction failed: {str(e)}"
        print(f"\n❌ ERROR: {error_msg}")
        print(traceback.format_exc())
        return returnFormat('error', error_msg)