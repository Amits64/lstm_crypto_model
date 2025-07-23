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


# Global variables for model and scaler
model = None
scaler = None
y_scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_in_progress = False


def load_model_and_scaler():
    """Load the trained model and scaler"""
    global model, scaler, y_scaler

    try:
        # Load scaler
        if os.path.exists('model/scaler.pkl'):
            scaler = joblib.load('model/scaler.pkl')
            print("Scaler loaded successfully")
        else:
            print("Scaler not found, creating new one")
            scaler = MinMaxScaler()

        # Load y_scaler
        if os.path.exists('model/y_scaler.pkl'):
            y_scaler = joblib.load('model/y_scaler.pkl')
            print("Y-scaler loaded successfully")
        else:
            print("Y-scaler not found")
            y_scaler = None

        # Load model
        if os.path.exists('model/trained_model.pt'):
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

            model = CNNLSTMModel(**model_params)
            model.load_state_dict(torch.load('model/trained_model.pt', map_location=device))
            model.to(device)
            model.eval()
            print("Model loaded successfully")
        else:
            print("Trained model not found")

    except Exception as e:
        print(f"Error loading model/scaler: {e}")


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
        if scaler is not None:
            scaled_data = scaler.transform(feature_data)
        else:
            # If scaler not available, use the last sequence as is (normalized)
            scaled_data = (feature_data - feature_data.mean()) / feature_data.std()
            scaled_data = scaled_data.fillna(0).values

        # Get the last sequence for prediction
        sequence = scaled_data[-sequence_length:]
        return np.expand_dims(sequence, axis=0)  # Add batch dimension

    except Exception as e:
        print(f"Error preparing data: {e}")
        return None


def make_prediction(data):
    """Make price prediction using the trained model"""
    try:
        if model is None:
            return None

        # Prepare data
        input_data = prepare_data_for_prediction(data)
        if input_data is None:
            return None

        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).to(device)

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor)
            predicted_price = prediction.cpu().numpy()[0][0]

        # If y_scaler is available, inverse transform the prediction
        if y_scaler is not None:
            try:
                predicted_price_reshaped = np.array([[predicted_price]])
                inverse_scaled = y_scaler.inverse_transform(predicted_price_reshaped)
                predicted_price = inverse_scaled[0][0]
            except Exception as e:
                print(f"Error inverse transforming prediction: {e}")
                # Use current price as fallback
                predicted_price = data[-1]['close']

        return float(predicted_price)

    except Exception as e:
        print(f"Error making prediction: {e}")
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
            training_in_progress = True
            print(f"Starting background training for {symbol} at {interval} interval...")

            # Import and run training (inline to avoid import issues)
            from model.train_model import train_model
            result = train_model(symbol=symbol, interval=interval, epochs=50, batch_size=32)

            if result.get('status') == 'success':
                print("Training completed successfully!")
                # Reload the model after training
                load_model_and_scaler()
            else:
                print(f"Training failed: {result}")

        except Exception as e:
            print(f"Background training error: {e}")
        finally:
            training_in_progress = False

    # Start training in background thread
    training_thread = threading.Thread(target=training_task)
    training_thread.daemon = True
    training_thread.start()


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
    """Generate AI predictions for the selected symbol"""
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

        # Make AI prediction
        predicted_price = make_prediction(data)

        # Calculate technical indicators
        indicators = calculate_technical_indicators(data)

        # Calculate prediction confidence and direction
        if predicted_price is not None and predicted_price != current_price:
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100
            direction = "UP" if price_change > 0 else "DOWN"
            confidence = min(95, abs(price_change_percent) * 10)  # Simple confidence calculation
        else:
            predicted_price = current_price
            price_change = 0
            price_change_percent = 0
            direction = "NEUTRAL"
            confidence = 0

        return jsonify({
            'symbol': symbol,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_percent': price_change_percent,
            'direction': direction,
            'confidence': confidence,
            'indicators': indicators,
            'model_loaded': model is not None,
            'training_in_progress': training_in_progress,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/train')
def train_model_endpoint():
    """Endpoint to trigger model training"""
    global training_in_progress

    if training_in_progress:
        return jsonify({
            'status': 'info',
            'message': 'Training already in progress',
            'training_in_progress': True
        })

    try:
        symbol = request.args.get("symbol", "BTCUSDT")
        interval = request.args.get("interval", "5m")

        # Start training in background
        run_training_in_background(symbol=symbol, interval=interval)

        return jsonify({
            'status': 'success',
            'message': f'Model training started in background for {symbol} ({interval})',
            'training_in_progress': True,
            'symbol': symbol,
            'interval': interval
        })

    except Exception as e:
        training_in_progress = False
        return jsonify({
            'status': 'error',
            'message': f'Failed to start training: {str(e)}',
            'training_in_progress': False
        })


@app.route('/training_status')
def training_status():
    """Check training status"""
    return jsonify({
        'training_in_progress': training_in_progress,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


# Initialize model and scaler on startup
load_model_and_scaler()

if __name__ == '__main__':
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)

    # Run with specific configuration to avoid auto-reload issues
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)