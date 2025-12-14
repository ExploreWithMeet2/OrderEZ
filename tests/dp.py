"""
Complete test script for Dynamic Pricing System

This script demonstrates the entire workflow:
1. Train a model
2. Generate predictions
3. View results
4. Apply prices

Usage:
    python test_dp_system.py
"""

import asyncio
import httpx
from datetime import datetime

BASE_URL = "http://localhost:5000"

# Use a branch ID from your seeded data
BRANCH_ID = "jh70ns6tjkyrj3n0va5s292c017vdq98"


async def test_health():
    """Test if API is running"""
    print("\n" + "="*70)
    print("TESTING API HEALTH")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/dp/health")
        data = response.json()
        
        print(f"Status: {data['status']}")
        print(f"Models Directory: {data['models_directory']}")
        print(f"Models Exist: {data['models_exist']}")
        
        return data['status'] == 'healthy'


async def test_get_branches():
    """Get all branches"""
    print("\n" + "="*70)
    print("FETCHING BRANCHES")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/dp/branches")
        data = response.json()
        
        if data['status'] == 'success':
            branches = data['branches']
            print(f"Found {len(branches)} branches:")
            for branch in branches[:5]:  # Show first 5
                print(f"  - {branch['name']} ({branch['_id']})")
            
            if branches:
                global BRANCH_ID
                BRANCH_ID = branches[0]['_id']
                print(f"\nUsing branch: {BRANCH_ID}")
            
            return branches
        else:
            print("Failed to fetch branches")
            return []


async def test_train_model():
    """Train a model for the branch"""
    print("\n" + "="*70)
    print(f"TRAINING MODEL FOR BRANCH: {BRANCH_ID}")
    print("="*70)
    print("This may take 2-5 minutes...")
    
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{BASE_URL}/dp/train",
            json={"branch_id": BRANCH_ID}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Training successful!")
            print(f"Message: {data['message']}")
            
            metrics = data['data']['metrics']
            print("\nMetrics:")
            print(f"  MAE:      {metrics['mae']:.4f}%")
            print(f"  RMSE:     {metrics['rmse']:.4f}%")
            print(f"  R²:       {metrics['r2']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            
            return True
        else:
            print(f"\n❌ Training failed: {response.text}")
            return False


async def test_get_model_info():
    """Get model information"""
    print("\n" + "="*70)
    print("FETCHING MODEL INFORMATION")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/dp/model/{BRANCH_ID}")
        data = response.json()
        
        if data['status'] == 'success':
            model = data['model']
            print(f"Model Version: {model['modelVersion']}")
            print(f"Trained At: {datetime.fromtimestamp(model['trainedAt']/1000)}")
            print(f"Accuracy: {model['accuracy']:.2f}%")
            print(f"MAE: {model['mae']:.4f}")
            print(f"Total Samples: {int(model['totalSamples'])}")
            print(f"Active: {model['isActive']}")
            return model
        else:
            print(f"Status: {data['status']}")
            print(f"Message: {data.get('message', 'No message')}")
            return None


async def test_predict_prices():
    """Generate price predictions"""
    print("\n" + "="*70)
    print("GENERATING PRICE PREDICTIONS")
    print("="*70)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/dp/predict",
            json={"branch_id": BRANCH_ID}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Predictions generated!")
            print(f"Message: {data['message']}")
            
            summary = data['data']['summary']
            print("\nSummary:")
            print(f"  Total Items:    {summary['total']}")
            print(f"  High Demand:    {summary['high_demand']}")
            print(f"  Normal Demand:  {summary['normal_demand']}")
            print(f"  Low Demand:     {summary['low_demand']}")
            
            # Show sample predictions
            predictions = data['data']['predictions'][:5]
            print("\nSample Predictions:")
            print(f"{'Item':<40} {'Current':<10} {'Suggested':<10} {'Change':<10} {'Demand':<10} {'Action'}")
            print("-" * 90)
            
            for pred in predictions:
                item_id = pred['item_id'][:8] + "..."
                current = f"₹{pred['current_price']:.2f}"
                suggested = f"₹{pred['suggested_price']:.2f}"
                change = f"{pred['predicted_change_percent']:+.1f}%"
                demand = pred['demand_category']
                action = pred['recommendation']['action']
                
                print(f"{item_id:<40} {current:<10} {suggested:<10} {change:<10} {demand:<10} {action}")
            
            return predictions
        else:
            print(f"\n❌ Prediction failed: {response.text}")
            return []


async def test_get_predictions():
    """Get stored predictions"""
    print("\n" + "="*70)
    print("FETCHING STORED PREDICTIONS")
    print("="*70)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/dp/predictions/{BRANCH_ID}?limit=10")
        data = response.json()
        
        if data['status'] == 'success':
            predictions = data['predictions']
            print(f"Found {len(predictions)} predictions")
            
            # Show high demand items
            high_demand = [p for p in predictions if p['demandCategory'] == 'HIGH']
            if high_demand:
                print("\nHigh Demand Items (Price Increase Recommended):")
                for pred in high_demand[:3]:
                    print(f"  - Item: {pred['itemId'][:12]}...")
                    print(f"    Current: ₹{pred['currentPrice']:.2f}")
                    print(f"    Suggested: ₹{pred['suggestedPrice']:.2f}")
                    print(f"    Confidence: {pred['confidence']:.2%}")
                    print(f"    Reason: {pred['recommendation']['reason']}")
                    print()
            
            return predictions
        else:
            print(f"Failed to fetch predictions: {data}")
            return []


async def test_apply_price():
    """Test applying a price (optional)"""
    print("\n" + "="*70)
    print("TESTING PRICE APPLICATION")
    print("="*70)
    
    # Get a prediction first
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/dp/predictions/{BRANCH_ID}?limit=1")
        data = response.json()
        
        if data['status'] == 'success' and data['predictions']:
            pred = data['predictions'][0]
            
            print(f"Selected prediction:")
            print(f"  Item ID: {pred['itemId']}")
            print(f"  Current Price: ₹{pred['currentPrice']:.2f}")
            print(f"  Suggested Price: ₹{pred['suggestedPrice']:.2f}")
            
            # Ask for confirmation (in real scenario)
            print("\n⚠️  Would apply price in production")
            print("    (Skipping actual application in test)")
            
            # Uncomment to actually apply:
            # response = await client.post(
            #     f"{BASE_URL}/dp/apply-price",
            #     json={
            #         "item_id": pred['itemId'],
            #         "new_price": pred['suggestedPrice'],
            #         "prediction_id": pred['_id']
            #     }
            # )
            # apply_data = response.json()
            # print(f"\n✅ {apply_data['message']}")
            
            return True
        else:
            print("No predictions available to apply")
            return False


async def main():

    try:
        # Test 1: Health check
        healthy = await test_health()
        if not healthy:
            print("\n API is not healthy. Please start the server first.")
            return
        
        # Test 2: Get branches
        branches = await test_get_branches()
        if not branches:
            print("\n No branches found. Please seed the database first.")
            return
        
        # Test 3: Get model info (might not exist yet)
        model = await test_get_model_info()
        
        if not model:
            print("\n No model found. Training new model...")
            # Test 4: Train model
            trained = await test_train_model()
            if not trained:
                print("\n Training failed. Please check the logs.")
                return
            
            # Get model info again
            await test_get_model_info()
        else:
            print("\n Model already exists. Skipping training.")
        
        # Test 5: Generate predictions
        predictions = await test_predict_prices()
        if not predictions:
            print("\n Prediction failed. Please check the logs.")
            return
        
        # Test 6: Get stored predictions
        await test_get_predictions()
        
        # Test 7: Apply price (optional)
        await test_apply_price()

        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())