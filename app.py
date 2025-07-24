# app.py
from flask import Flask, render_template, request, jsonify
from binance.client import Client
from datetime import datetime, timedelta
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
import sys

executor = ThreadPoolExecutor(max_workers=1)  # Single-threaded for safety
traceback.print_exc()  # Print full error details
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Initialize Binance client
client = Client()


# Handle different versions of python-binance
def get_binance_interval(interval_str):
    """Get Binance interval constant, with fallback for different library versions"""
    interval_map = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M"
    }

    # Try to use constants first
    try:
        constants = {
            "1m": Client.KLINE_INTERVAL_1MINUTE,
            "3m": Client.KLINE_INTERVAL_3MINUTE,
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "30m": Client.KLINE_INTERVAL_30MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "2h": Client.KLINE_INTERVAL_2HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "6h": Client.KLINE_INTERVAL_6HOUR,
            "8h": Client.KLINE_INTERVAL_8HOUR,
            "12h": Client.KLINE_INTERVAL_12HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY,
            "3d": Client.KLINE_INTERVAL_3DAY,
            "1w": Client.KLINE_INTERVAL_1WEEK,
            "1M": Client.KLINE_INTERVAL_1MONTH
        }
        return constants.get(interval_str, interval_str)
    except AttributeError:
        # Fallback to string if constants don't exist
        return interval_map.get(interval_str, interval_str)


INTERVALS = {
    "1m": get_binance_interval("1m"),
    "3m": get_binance_interval("3m"),
    "5m": get_binance_interval("5m"),
    "15m": get_binance_interval("15m"),
    "30m": get_binance_interval("30m"),
    "1h": get_binance_interval("1h"),
    "2h": get_binance_interval("2h"),
    "4h": get_binance_interval("4h"),
    "6h": get_binance_interval("6h"),
    "8h": get_binance_interval("8h"),
    "12h": get_binance_interval("12h"),
    "1d": get_binance_interval("1d"),
    "3d": get_binance_interval("3d"),
    "1w": get_binance_interval("1w"),
    "1M": get_binance_interval("1M")
}

# Expanded symbol list with popular trading pairs
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT',
    'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'FILUSDT',
    'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'VETUSDT', 'ICPUSDT',
    'FTMUSDT', 'HBARUSDT', 'NEARUSDT', 'ALGOUSDT', 'QNTUSDT'
]


# CNN-LSTM Model Definition
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=5, cnn_filters=48, kernel_size=5,
                 lstm_units=128, lstm_units2=32, dense_units=64,
                 dropout_lstm=0.2, dropout_dense=0.3, dropout_dense2=0.3):
        super(CNNLSTMModel, self).__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(0.2)

        # LSTM layers
        self.lstm1 = nn.LSTM(cnn_filters * 2, lstm_units, batch_first=True, dropout=dropout_lstm)
        self.lstm2 = nn.LSTM(lstm_units, lstm_units2, batch_first=True, dropout=dropout_lstm)

        # Dense layers
        self.dropout_lstm = nn.Dropout(dropout_lstm)
        self.dense1 = nn.Linear(lstm_units2, dense_units)
        self.dropout_dense = nn.Dropout(dropout_dense)
        self.dense2 = nn.Linear(dense_units, 32)
        self.dropout_dense2 = nn.Dropout(dropout_dense2)
        self.output = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN forward pass
        x = x.transpose(1, 2)  # Change to (batch, features, sequence)
        x = self.relu(self.conv1(x))
        x = self.dropout_cnn(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # Transpose back for LSTM
        x = x.transpose(1, 2)  # Change to (batch, sequence, features)

        # LSTM forward pass
        x, _ = self.lstm1(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        x = self.dropout_lstm(x)

        # Take the last output
        x = x[:, -1, :]

        # Dense layers
        x = self.relu(self.dense1(x))
        x = self.dropout_dense(x)
        x = self.relu(self.dense2(x))
        x = self.dropout_dense2(x)
        x = self.output(x)

        return x


# Global variables for model and scaler - now per interval
models = {}  # Dictionary to store models by interval
scalers = {}  # Dictionary to store scalers by interval
y_scalers = {}  # Dictionary to store y_scalers by interval
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_in_progress = {}  # Track training status per interval


def get_model_path(interval):
    """Get model file paths for a specific interval"""
    return {
        'model': f'model/models/{interval}/trained_model.pt',
        'scaler': f'model/models/{interval}/scaler.pkl',
        'y_scaler': f'model/models/{interval}/y_scaler.pkl',
        'metadata': f'model/models/{interval}/training_metadata.pkl'
    }


def load_model_and_scaler(interval='5m'):
    """Load the trained model and scaler for specific interval"""
    global models, scalers, y_scalers

    try:
        paths = get_model_path(interval)

        # Load scaler
        if os.path.exists(paths['scaler']):
            scalers[interval] = joblib.load(paths['scaler'])
            print(f"Scaler loaded successfully for {interval}")
        else:
            print(f"Scaler not found for {interval}, creating new one")
            scalers[interval] = MinMaxScaler()

        # Load y_scaler
        if os.path.exists(paths['y_scaler']):
            y_scalers[interval] = joblib.load(paths['y_scaler'])
            print(f"Y-scaler loaded successfully for {interval}")
        else:
            print(f"Y-scaler not found for {interval}")
            y_scalers[interval] = None

        # Load model
        if os.path.exists(paths['model']):
            # Best hyperparameters
            model_params = {
                'cnn_filters': 48,
                'kernel_size': 5,
                'lstm_units': 128,
                'dropout_lstm': 0.2,
                'lstm_units2': 32,
                'dropout_dense': 0.3,
                'dense_units': 64,
                'dropout_dense2': 0.3
            }

            models[interval] = CNNLSTMModel(**model_params)
            models[interval].load_state_dict(torch.load(paths['model'], map_location=device))
            models[interval].to(device)
            models[interval].eval()
            print(f"Model loaded successfully for {interval}")
        else:
            print(f"Trained model not found for {interval}")
            models[interval] = None

    except Exception as e:
        print(f"Error loading model/scaler for {interval}: {e}")


def prepare_data_for_prediction(data, sequence_length=60):
    """Prepare data for model prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_ma'] = df['volume'].rolling(window=5).mean()

        # Select features for prediction
        features = ['open', 'high', 'low', 'close', 'volume']
        feature_data = df[features].fillna(method='ffill').fillna(0)

        if len(feature_data) < sequence_length:
            return None

        # Scale the data
        return feature_data, features

    except Exception as e:
        print(f"Error preparing data: {e}")
        return None, None


def make_prediction(data, interval='5m'):
    """Make price prediction using the trained model for a specific interval"""
    try:
        if interval not in models or models[interval] is None:
            print(f"No model loaded for interval {interval}")
            return None

        # Prepare data
        feature_data, features = prepare_data_for_prediction(data)
        if feature_data is None:
            return None

        # Scale the data using interval-specific scaler
        if interval in scalers and scalers[interval] is not None:
            scaled_data = scalers[interval].transform(feature_data)
        else:
            # If scaler not available, use the last sequence as is (normalized)
            scaled_data = (feature_data - feature_data.mean()) / feature_data.std()
            scaled_data = scaled_data.fillna(0).values

        # Get the last sequence for prediction
        sequence = scaled_data[-60:]  # sequence_length = 60
        input_data = np.expand_dims(sequence, axis=0)  # Add batch dimension

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = models[interval](input_tensor)
            predicted_price = prediction.cpu().numpy()[0][0]

        # If y_scaler is available, inverse transform the prediction
        if interval in y_scalers and y_scalers[interval] is not None:
            try:
                predicted_price_reshaped = np.array([[predicted_price]])
                inverse_scaled = y_scalers[interval].inverse_transform(predicted_price_reshaped)
                predicted_price = inverse_scaled[0][0]
            except Exception as e:
                print(f"Error inverse transforming prediction: {e}")
                # Use current price as fallback
                predicted_price = data[-1]['close']

        return float(predicted_price)

    except Exception as e:
        print(f"Error making prediction for {interval}: {e}")
        return None


def calculate_technical_indicators(data):
    """Calculate additional technical indicators"""
    try:
        df = pd.DataFrame(data)

        # Moving averages
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        return {
            'ma_20': df['ma_20'].iloc[-1] if not df['ma_20'].isna().iloc[-1] else None,
            'ma_50': df['ma_50'].iloc[-1] if not df['ma_50'].isna().iloc[-1] else None,
            'rsi': df['rsi'].iloc[-1] if not df['rsi'].isna().iloc[-1] else None,
            'macd': df['macd'].iloc[-1] if not df['macd'].isna().iloc[-1] else None,
            'macd_signal': df['macd_signal'].iloc[-1] if not df['macd_signal'].isna().iloc[-1] else None
        }
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return {}


def run_training_in_background(symbol='BTCUSDT', interval='5m'):
    """Run training in a separate thread to avoid blocking the main app"""
    global training_in_progress

    def training_task():
        global training_in_progress
        try:
            training_in_progress[interval] = True
            print(f"🔥 BACKGROUND TRAINING STARTED: {symbol} at {interval} interval")
            print(f"📊 Training parameters: symbol={symbol}, interval={interval}")

            # Add the models directory to a Python path so we can import train_model
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            if models_dir not in sys.path:
                sys.path.append(models_dir)

            # Import and run training
            from model.train_model import train_model

            # Create interval-specific directory
            interval_dir = f'model/models/{interval}'
            os.makedirs(interval_dir, exist_ok=True)
            print(f"📁 Created directory: {interval_dir}")

            # Train the model with explicit parameters
            print(f"🚀 Calling train_model with: symbol={symbol}, interval={interval}")
            result = train_model(symbol=symbol, interval=interval, epochs=50, batch_size=32)

            if result.get('status') == 'success':
                print(f"✅ Training completed successfully for {symbol} - {interval}!")

                # Move model files to interval-specific directory
                try:
                    file_moves = [
                        ('model/trained_model.pt', f'{interval_dir}/trained_model.pt'),
                        ('model/scaler.pkl', f'{interval_dir}/scaler.pkl'),
                        ('model/y_scaler.pkl', f'{interval_dir}/y_scaler.pkl'),
                        ('model/training_metadata.pkl', f'{interval_dir}/training_metadata.pkl')
                    ]

                    for src, dst in file_moves:
                        if os.path.exists(src):
                            os.replace(src, dst)
                            print(f"📦 Moved: {src} → {dst}")
                        else:
                            print(f"⚠️ File not found: {src}")

                    print(f"🎉 Model files successfully moved to {interval_dir}")

                    # Reload the model for this interval
                    load_model_and_scaler(interval)
                    print(f"🔄 Reloaded model for {interval}")

                except Exception as move_error:
                    print(f"❌ Error moving model files: {move_error}")
            else:
                print(f"❌ Training failed for {interval}: {result}")

        except Exception as e:
            print(f"💥 Background training error for {interval}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            training_in_progress[interval] = False
            print(f"🏁 Training process finished for {interval}")

    # Start training in background thread
    training_thread = threading.Thread(target=training_task)
    training_thread.daemon = True
    training_thread.start()
    print(f"🧵 Background thread started for {symbol} - {interval}")


@app.route('/')
def index():
    return render_template('index.html', symbols=SYMBOLS)


@app.route('/candles')
def candles():
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "5m")
    binance_interval = INTERVALS.get(interval, "5m")

    # Increased limit for better scrolling experience
    limit = 1000
    try:
        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=limit)
        data = [{
            'time': int(k[0]),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        } for k in klines]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict')
def predict():
    """Generate AI predictions for the selected symbol and interval"""
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "5m")
    binance_interval = INTERVALS.get(interval, "5m")

    try:
        # Get recent data for prediction
        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=100)
        data = [{
            'time': int(k[0]),
            'open': float(k[1]),
            'high': float(k[2]),
            'low': float(k[3]),
            'close': float(k[4]),
            'volume': float(k[5])
        } for k in klines]

        current_price = data[-1]['close']

        # Make AI prediction using interval-specific model
        predicted_price = make_prediction(data, interval)

        # Calculate technical indicators
        indicators = calculate_technical_indicators(data)

        # Calculate prediction confidence and direction
        if predicted_price is not None and predicted_price != current_price:
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            direction = "UP" if price_change > 0 else "DOWN"
            confidence = min(95, abs(int(price_change_percent * 10)))  # Simple confidence calculation
        else:
            predicted_price = current_price
            price_change = 0
            price_change_percent = 0
            direction = "NEUTRAL"
            confidence = 0

        return jsonify({
            'symbol': symbol,
            'interval': interval,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'direction': direction,
            'confidence': confidence,
            'indicators': indicators,
            'model_loaded': interval in models and models[interval] is not None,
            'training_in_progress': training_in_progress.get(interval, False),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/train')
def train_model_endpoint():
    """Endpoint to trigger model training for specific interval"""
    global training_in_progress

    # Get parameters from request
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "5m")

    print(f"🎯 Training request received: symbol={symbol}, interval={interval}")

    if training_in_progress.get(interval, False):
        return jsonify({
            'status': 'info',
            'message': f'Training already in progress for {interval}',
            'training_in_progress': True,
            'interval': interval
        })

    try:
        # Start training in background
        print(f"🚀 Starting training for {symbol} at {interval} interval...")
        run_training_in_background(symbol=symbol, interval=interval)

        return jsonify({
            'status': 'success',
            'message': f'Model training started in background for {symbol} ({interval})',
            'training_in_progress': True,
            'symbol': symbol,
            'interval': interval
        })

    except Exception as e:
        training_in_progress[interval] = False
        return jsonify({
            'status': 'error',
            'message': f'Failed to start training: {str(e)}',
            'training_in_progress': False,
            'interval': interval
        })


@app.route('/training_status')
def training_status():
    """Check training status for a specific interval"""
    interval = request.args.get("interval", "5m")

    return jsonify({
        'interval': interval,
        'training_in_progress': training_in_progress.get(interval, False),
        'model_loaded': interval in models and models[interval] is not None,
        'scaler_loaded': interval in scalers and scalers[interval] is not None
    })


# Initialize models and scalers on startup for common intervals
common_intervals = ['1m', '5m', '15m', '1h', '4h', '1d']
for interval in common_intervals:
    if os.path.exists(f'model/models/{interval}'):
        load_model_and_scaler(interval)

if __name__ == '__main__':
    # Ensure model directories exist
    os.makedirs('model', exist_ok=True)
    for interval in common_intervals:
        os.makedirs(f'model/models/{interval}', exist_ok=True)

    # Run with specific configuration to avoid auto-reload issues
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)