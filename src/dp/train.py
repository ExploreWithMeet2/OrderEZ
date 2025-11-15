from datetime import datetime
from pathlib import Path
import pickle
from typing import Dict, Tuple
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from core.config import Config
from src.dp.convex_caller.main import fetch_training_data, store_model_metadata
from utils.returnFormat import returnFormat
from src.dp.preprocessing import (
    dp_preprocessing,
    prepare_sequences,
)

MODEL_DIR = Path("models")
SEQUENCE_LENGTH = 30
EPOCHS = 100
BATCH_SIZE = 32
TEST_SIZE = 0.2
EARLY_STOP_PATIENCE = 15

MAX_PRICE_CHANGE_PERCENT = 20.0
MIN_PRICE_CHANGE_PERCENT = -20.0

TRAINING_DAYS = 90
MIN_CONFIDENCE = 0.6

b_a = Config.b_a
def build_model(input_shape: Tuple[int, int]) -> Sequential:
    
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16),
        LeakyReLU(alpha=0.1),
        Dropout(0.2),
        
        Dense(8),
        LeakyReLU(alpha=0.1),
        
        Dense(1, activation='linear')
    ])
    
    opt = Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    print(f"\nModel built: {model.count_params():,} parameters")
    return model


def prepare_training_data(
    processed_data: pd.DataFrame,
    branch_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"\nPreparing training data for branch {branch_id}...")
    X, y = prepare_sequences(
        processed_data, 
        sequence_length=SEQUENCE_LENGTH,
        target_col='price_change_percent'
    )
    
    if len(X) == 0:
        raise ValueError("No sequences created - insufficient data")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=42,
        shuffle=True
    )
    
    print(f"  Training set: {len(X_train)} sequences")
    print(f"  Validation set: {len(X_val)} sequences")
    
    return X_train, X_val, y_train, y_val


def evaluate_model(
    model: Sequential,
    X_val: np.ndarray,
    y_val: np.ndarray,
    branch_id: str
) -> Dict:
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_val, verbose=0).flatten()
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    
    mape = np.mean(np.abs((y_val - y_pred) / (np.abs(y_val) + 1e-10))) * 100
    
    accuracy = np.mean(np.abs(y_val - y_pred) <= 2) * 100
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape),
        'accuracy': float(accuracy),
        'trained_at': datetime.now().timestamp() * 1000,
        'total_samples': int(len(y_val)),
        'branch_id': branch_id,
        'sequence_length': SEQUENCE_LENGTH
    }
    
    print("MODEL PERFORMANCE METRICS")
    print(f"Mean Absolute Error (MAE):           {mae:.4f}%")
    print(f"Root Mean Squared Error (RMSE):      {rmse:.4f}%")
    print(f"R² Score:                            {r2:.4f}")
    print(f"Mean Absolute Percentage Error:      {mape:.2f}%")
    print(f"Prediction Accuracy (±2%):           {accuracy:.2f}%")
    
    print(f"{'Actual %':<12} {'Predicted %':<14} {'Difference':<12} {'Within ±2%'}")
    
    for i in range(min(10, len(y_val))):
        diff = y_pred[i] - y_val[i]
        within_threshold = "✓" if abs(diff) <= 2 else "✗"
        print(f"{y_val[i]:>6.2f}%      {y_pred[i]:>6.2f}%         "
              f"{diff:>+6.2f}%       {within_threshold}")
    
    return metrics


def save_model_artifacts(
    model: Sequential,
    branch_id: str,
    metrics: Dict,
    training_history: Dict
):
    branch_model_dir = MODEL_DIR / branch_id
    branch_model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = branch_model_dir / "model.h5"
    model.save(model_path)
    
    preprocessor_path = branch_model_dir / "preprocessor.pkl"
    
    metadata = {
        'branch_id': branch_id,
        'sequence_length': SEQUENCE_LENGTH,
        'metrics': metrics,
        'training_history': {
            k: [float(v) for v in vals] 
            for k, vals in training_history.items()
        },
        'trained_at': datetime.now().isoformat(),
        'trained_at_ms': datetime.now().timestamp() * 1000,
        'model_path': str(model_path),
        'preprocessor_path': str(preprocessor_path),
        'model_version': 'v1.0',
        'max_price_change': MAX_PRICE_CHANGE_PERCENT,
        'min_price_change': MIN_PRICE_CHANGE_PERCENT
    }
    
    metadata_path = branch_model_dir / "metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    return metadata


async def train(branch_id: str) -> dict:
    try:
        for b in b_a:
            print(f"Training BranchId: {b}")
        
        data_resp = await fetch_training_data(branch_id, days=TRAINING_DAYS)
        
        if data_resp['type'] == 'error':
            return returnFormat('error', data_resp['message'])
        
        training_data = data_resp['data']
        
        if len(training_data) < 100:
            return returnFormat(
                'error',
                f"Insufficient data: {len(training_data)} records (need at least 100)"
            )
        
        
        df = pd.DataFrame(training_data)
        
        if 'dt' in df.columns and isinstance(df['dt'].iloc[0], str):
            df['dt'] = pd.to_datetime(df['dt'])
        
        preprocess_result = dp_preprocessing(df, branch_id, mode="train")
        
        if preprocess_result['type'] == 'error':
            return preprocess_result
        
        processed_data = preprocess_result['data']
        
        X_train, X_val, y_train, y_val = prepare_training_data(processed_data, branch_id)
        
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOP_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        training_history = history.history
        
        metrics = evaluate_model(model, X_val, y_val, branch_id)
        
        metadata = save_model_artifacts(model, branch_id, metrics, training_history)
        
        await store_model_metadata(branch_id, metadata)
        
        return returnFormat(
            'success',
            f"Model trained successfully with {metrics['accuracy']:.2f}% accuracy",
            {
                'branch_id': branch_id,
                'metrics': metrics,
                'model_path': str(metadata['model_path']),
                'total_sequences': len(X_train) + len(X_val)
            }
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}"
        print(f"\nERROR: {error_msg}")
        print(traceback.format_exc())
        return returnFormat('error', error_msg)