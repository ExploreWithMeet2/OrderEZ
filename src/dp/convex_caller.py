from datetime import datetime, timedelta
from typing import Dict, List, Optional

from app.schema.convex_schema import ConvexRequest
from app.utils.convex import call_convex
from app.utils.returnFormat import returnFormat


async def fetch_branch_items(branch_id: str) -> dict:
    req = ConvexRequest(
        module="items",
        func="getByBranch",
        isQuery=True,
        args={"branchId": branch_id},
        returnDf=False,
    )
    return await call_convex(req)


async def fetch_price_history(item_id: str, days: int = 30) -> dict:
    end_time = datetime.now().timestamp() * 1000
    start_time = (datetime.now() - timedelta(days=days)).timestamp() * 1000

    req = ConvexRequest(
        module="prices",
        func="getHistoryByItem",
        isQuery=True,
        args={"itemId": item_id, "startTime": start_time, "endTime": end_time},
        returnDf=False,
    )
    return await call_convex(req)


async def fetch_item_metrics(item_id: str, days: int = 7) -> dict:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    req = ConvexRequest(
        module="itemMetrics",
        func="getMetrics",
        isQuery=True,
        args={"itemId": item_id, "startDate": start_date, "endDate": end_date},
        returnDf=False,
    )
    return await call_convex(req)


async def fetch_orders_by_branch(branch_id: str, days: int = 30) -> dict:
    end_time = datetime.now().timestamp() * 1000
    start_time = (datetime.now() - timedelta(days=days)).timestamp() * 1000

    req = ConvexRequest(
        module="orders",
        func="getByBranchTimeRange",
        isQuery=True,
        args={"branchId": branch_id, "startTime": start_time, "endTime": end_time},
        returnDf=False,
    )
    return await call_convex(req)


async def fetch_training_data(branch_id: str, days: int = 90) -> dict:
    items_response = await fetch_branch_items(branch_id)
    if items_response["type"] == "error":
        return items_response

    items_list = items_response["data"]
    if not items_list:
        return returnFormat("error", f"No items found for branch {branch_id}")

    print(f"Found {len(items_list)} items")

    all_data = []

    for item in items_list:
        item_id = item["_id"]

        prices_response = await fetch_price_history(item_id, days)
        if prices_response["type"] == "error":
            continue

        prices_list = prices_response["data"]
        if not prices_list:
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
                "dt": dt.isoformat(),
                "demand_7d": price_row.get("demand", "MEDIUM"),
                "rating_7d": 4,
                "orders_7d": metrics.get("totalOrders", 0),
                "revenue_7d": metrics.get("totalRevenue", 0),
                "avg_quantity": metrics.get("totalQuantity", 0)
                / max(metrics.get("totalOrders", 1), 1),
                "time_of_day": (
                    "NOON"
                    if 12 <= dt.hour < 15
                    else (
                        "MORNING"
                        if 5 <= dt.hour < 12
                        else "AFTERNOON" if 15 <= dt.hour < 20 else "NIGHT"
                    )
                ),
                "season": (
                    "WINTER"
                    if dt.month in [12, 1, 2]
                    else "SUMMER" if dt.month in [3, 4, 5, 6] else "MONSOON"
                ),
                "day_of_week": dt.weekday(),
                "is_weekend": 1 if dt.weekday() >= 5 else 0,
                "is_holiday": price_row.get("isEvent", False),
                "is_event": price_row.get("isEvent", False),
                "event_name": "",
            }
            all_data.append(record)

    return returnFormat(
        "success", f"Training data fetched: {len(all_data)} records", all_data
    )


async def store_prediction(prediction: Dict) -> dict:
    req = ConvexRequest(
        module="pricingPredictions",
        func="create",
        isQuery=False,
        args={
            "itemId": prediction["item_id"],
            "branchId": prediction["branch_id"],
            "timestamp": prediction["predicted_at"],
            "predictedChangePercent": prediction["predicted_change_percent"],
            "confidence": prediction["confidence"],
            "currentPrice": prediction["current_price"],
            "suggestedPrice": prediction["suggested_price"],
            "demandCategory": prediction["demand_category"],
            "isApplied": False,
            "recommendation": prediction["recommendation"],
        },
        returnDf=False,
    )
    return await call_convex(req)


async def store_predictions_batch(predictions: List[Dict]) -> dict:
    successful = 0
    failed = 0

    for pred in predictions:
        result = await store_prediction(pred)
        if result["type"] == "success":
            successful += 1
        else:
            failed += 1

    return returnFormat(
        "success" if failed == 0 else "error",
        f"Stored {successful} predictions, {failed} failed",
        {"successful": successful, "failed": failed},
    )


async def update_item_price(
    item_id: str, new_price: float, prediction_id: Optional[str] = None
) -> dict:
    req1 = ConvexRequest(
        module="items",
        func="updatePrice",
        isQuery=False,
        args={"itemId": item_id, "currentPrice": new_price},
        returnDf=False,
    )
    result1 = await call_convex(req1)

    if result1["type"] == "error":
        return result1

    req2 = ConvexRequest(
        module="prices",
        func="create",
        isQuery=False,
        args={
            "itemId": item_id,
            "value": new_price,
            "updatedAt": datetime.now().timestamp() * 1000,
        },
        returnDf=False,
    )

    if prediction_id:
        req3 = ConvexRequest(
            module="pricingPredictions",
            func="markAsApplied",
            isQuery=False,
            args={
                "predictionId": prediction_id,
                "appliedAt": datetime.now().timestamp() * 1000,
            },
            returnDf=False,
        )
        await call_convex(req3)

    return returnFormat("success", f"Price updated to ₹{new_price:.2f}")


async def store_model_metadata(branch_id: str, metadata: Dict) -> dict:
    req = ConvexRequest(
        module="mlModels",
        func="create",
        isQuery=False,
        args={
            "branchId": branch_id,
            "modelVersion": metadata.get("model_version", "v1.0"),
            "trainedAt": metadata["trained_at_ms"],
            "accuracy": metadata.get("accuracy", 0),
            "mae": metadata.get("mae", 0),
            "rmse": metadata.get("rmse", 0),
            "totalSamples": metadata.get("total_samples", 0),
            "modelPath": metadata.get("model_path", ""),
            "preprocessorPath": metadata.get("preprocessor_path", ""),
            "isActive": True,
            "sequenceLength": metadata.get("sequence_length"),
            "maxPriceChange": metadata.get("max_price_change"),
            "minPriceChange": metadata.get("min_price_change"),
        },
        returnDf=False,
    )
    return await call_convex(req)


async def get_all_branches() -> dict:
    req = ConvexRequest(
        module="branches", func="getAllActive", isQuery=True, args={}, returnDf=False
    )
    return await call_convex(req)


async def create_owner_alert(
    branch_id: str,
    item_id: str,
    alert_type: str,
    severity: str,
    message: str,
    predicted_change_percent: Optional[float] = None,
    confidence: Optional[float] = None,
) -> dict:
    req = ConvexRequest(
        module="ownerAlerts",
        func="create",
        isQuery=False,
        args={
            "branchId": branch_id,
            "itemId": item_id,
            "alertType": alert_type,
            "severity": severity,
            "message": message,
            "predictedChangePercent": predicted_change_percent,
            "confidence": confidence,
            "timestamp": datetime.now().timestamp() * 1000,
            "isRead": False,
        },
        returnDf=False,
    )
    return await call_convex(req)


async def get_unread_alerts(branch_id: str) -> dict:
    req = ConvexRequest(
        module="ownerAlerts",
        func="getUnread",
        isQuery=True,
        args={"branchId": branch_id},
        returnDf=False,
    )
    return await call_convex(req)


async def get_active_model(branch_id: str) -> dict:
    req = ConvexRequest(
        module="mlModels",
        func="getActive",
        isQuery=True,
        args={"branchId": branch_id},
        returnDf=False,
    )
    return await call_convex(req)


async def get_latest_predictions(branch_id: str, limit: int = 20) -> dict:
    req = ConvexRequest(
        module="pricingPredictions",
        func="getByBranch",
        isQuery=True,
        args={"branchId": branch_id, "limit": limit},
        returnDf=False,
    )
    return await call_convex(req)


async def get_predictions_by_demand(branch_id: str, demand_category: str) -> dict:
    req = ConvexRequest(
        module="pricingPredictions",
        func="getByDemandCategory",
        isQuery=True,
        args={"branchId": branch_id, "demandCategory": demand_category},
        returnDf=False,
    )
    return await call_convex(req)
