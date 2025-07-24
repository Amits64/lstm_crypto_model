# models/train_all_timeframes.py
import os

# Import everything from your full model
from train_model import train_model

# List of Binance time intervals you want to train for
timeframes = [
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h',
    '12h', '1d', '3d', '1w', '1M'
]

# Training loop for all intervals
def train_all_models(symbol='BTCUSDT', epochs=100, batch_size=32):
    for interval in timeframes:
        print(f"\n====== Starting training for interval: {interval} ======")

        # Set a unique model directory for each timeframe
        model_dir = f'models/{interval}'
        os.makedirs(model_dir, exist_ok=True)

        # Train the model
        result = train_model(
            symbol=symbol,
            epochs=epochs,
            batch_size=batch_size,
            interval=interval
        )

        # If training was successful, move model files to the appropriate directory
        if result.get("status") == "success":
            try:
                os.replace('model/trained_model.pt', f'{model_dir}/trained_model.pt')
                os.replace('model/scaler.pkl', f'{model_dir}/scaler.pkl')
                os.replace('model/y_scaler.pkl', f'{model_dir}/y_scaler.pkl')
                os.replace('model/training_metadata.pkl', f'{model_dir}/training_metadata.pkl')
                print(f"✅ Saved model and scalers to {model_dir}")
            except Exception as e:
                print(f"⚠️ Failed to move files for {interval}: {e}")
        else:
            print(f"❌ Training failed for interval {interval}: {result.get('error')}")

    print("\n✅ All models trained and saved.")


if __name__ == "__main__":
    train_all_models(
        symbol='BTCUSDT',  # You can change this to any symbol like 'ETHUSDT'
        epochs=100,
        batch_size=32
    )
