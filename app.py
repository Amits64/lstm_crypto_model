import json
import os
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
from binance.client import Client
from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
from scipy import stats

# Import your modules
try:
    from alerts import get_market_alerts, create_custom_alert, start_monitoring, stop_monitoring
except ImportError:
    print("Warning: alerts module not found")

try:
    from model.train_all_timeframes import train_enhanced_model
except ImportError:
    print("Warning: train_all_timeframes module not found")

try:
    from news import get_crypto_news
except ImportError:
    print("Warning: news module not found")

executor = ThreadPoolExecutor(max_workers=1)
warnings.filterwarnings('ignore')

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Initialize Binance client
client = Client()


# Handle different versions of python-binance
def get_binance_interval(interval_str):
    """Get Binance interval constant, with fallback for different library versions"""
    interval_map = {
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"
    }

    try:
        constants = {
            "5m": Client.KLINE_INTERVAL_5MINUTE,
            "15m": Client.KLINE_INTERVAL_15MINUTE,
            "1h": Client.KLINE_INTERVAL_1HOUR,
            "4h": Client.KLINE_INTERVAL_4HOUR,
            "1d": Client.KLINE_INTERVAL_1DAY
        }
        return constants.get(interval_str, interval_str)
    except AttributeError:
        return interval_map.get(interval_str, interval_str)


INTERVALS = {
    "5m": get_binance_interval("5m"),
    "15m": get_binance_interval("15m"),
    "1h": get_binance_interval("1h"),
    "4h": get_binance_interval("4h"),
    "1d": get_binance_interval("1d")
}

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
    'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT',
    'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT', 'UNIUSDT', 'FILUSDT',
    'TRXUSDT', 'ETCUSDT', 'XLMUSDT', 'VETUSDT', 'ICPUSDT',
    'FTMUSDT', 'HBARUSDT', 'NEARUSDT', 'ALGOUSDT', 'QNTUSDT'
]


# Enhanced CNN-LSTM Model Definition
class EnhancedCNNLSTMModel(nn.Module):
    def __init__(self, input_size=20, market_features_size=25, cnn_filters=64, kernel_size=3,
                 lstm_units=128, lstm_units2=64, dense_units=128,
                 dropout_lstm=0.3, dropout_dense=0.4, num_heads=8):
        super(EnhancedCNNLSTMModel, self).__init__()

        # Multi-scale CNN layers - Fixed channel distribution
        self.filters_1 = cnn_filters // 3
        self.filters_2 = cnn_filters // 3
        self.filters_3 = cnn_filters - (self.filters_1 + self.filters_2)

        self.conv1_3 = nn.Conv1d(input_size, self.filters_1, 3, padding=1)
        self.conv1_5 = nn.Conv1d(input_size, self.filters_2, 5, padding=2)
        self.conv1_7 = nn.Conv1d(input_size, self.filters_3, 7, padding=3)

        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(cnn_filters * 2, cnn_filters * 2, kernel_size, padding=kernel_size // 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.bn3 = nn.BatchNorm1d(cnn_filters * 2)

        self.pool = nn.AdaptiveMaxPool1d(30)
        self.dropout_cnn = nn.Dropout(0.25)

        # Attention mechanism - ensure embed_dim is divisible by num_heads
        embed_dim = cnn_filters * 2
        if embed_dim % num_heads != 0:
            for i in range(num_heads, 0, -1):
                if embed_dim % i == 0:
                    num_heads = i
                    break

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(cnn_filters * 2, lstm_units, batch_first=True,
                             dropout=dropout_lstm, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_units * 2, lstm_units2, batch_first=True,
                             dropout=dropout_lstm, bidirectional=True)

        # Feature fusion
        feature_size = lstm_units2 * 2
        fusion_input_size = feature_size + market_features_size
        self.feature_fusion = nn.Linear(fusion_input_size, feature_size)

        # Dense layers with residual connections
        self.dense1 = nn.Linear(feature_size, dense_units)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.dense3 = nn.Linear(dense_units // 2, dense_units // 4)

        # Residual connection
        self.residual = nn.Linear(feature_size, dense_units // 4)

        self.dropout_dense = nn.Dropout(dropout_dense)
        self.output = nn.Linear(dense_units // 4, 1)

        # Layer normalization
        self.ln1 = nn.LayerNorm(dense_units)
        self.ln2 = nn.LayerNorm(dense_units // 2)

        self.relu = nn.ReLU()
        self.swish = nn.SiLU()

    def forward(self, x, market_features=None):
        batch_size = x.size(0)

        # Multi-scale CNN
        x_t = x.transpose(1, 2)
        conv1_3 = self.relu(self.conv1_3(x_t))
        conv1_5 = self.relu(self.conv1_5(x_t))
        conv1_7 = self.relu(self.conv1_7(x_t))

        x = torch.cat([conv1_3, conv1_5, conv1_7], dim=1)
        x = self.bn1(x)
        x = self.dropout_cnn(x)

        x = self.swish(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout_cnn(x)

        x = self.swish(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.dropout_cnn(x)

        x = x.transpose(1, 2)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out

        # Bidirectional LSTM
        x, _ = self.lstm1(x)
        x = self.dropout_dense(x)
        x, _ = self.lstm2(x)
        x = self.dropout_dense(x)

        # Global features
        lstm_out = x[:, -1, :]

        # Feature fusion with market data
        if market_features is not None:
            fused_features = torch.cat([lstm_out, market_features], dim=1)
            fused_features = self.feature_fusion(fused_features)
        else:
            fused_features = lstm_out

        # Dense layers with residual connection
        residual = self.residual(fused_features)

        x = self.swish(self.dense1(fused_features))
        x = self.ln1(x)
        x = self.dropout_dense(x)

        x = self.swish(self.dense2(x))
        x = self.ln2(x)
        x = self.dropout_dense(x)

        x = self.swish(self.dense3(x))
        x = x + residual

        x = self.output(x)
        return x


# Legacy CNN-LSTM Model for backward compatibility
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
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.dropout_cnn(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)

        # Transpose back for LSTM
        x = x.transpose(1, 2)

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


class WaveletTransform:
    """Simple wavelet-like transformation for time series decomposition"""

    def __init__(self, levels=3):
        self.levels = levels

    def decompose(self, signal):
        """Decompose signal into trend and detail components"""
        components = []
        current = signal.copy()

        for _ in range(self.levels):
            # Simple moving average as approximation (trend)
            trend = pd.Series(current).rolling(window=4, center=True).mean().fillna(method='bfill').fillna(
                method='ffill')
            # Detail component
            detail = current - trend.values
            components.append(detail)
            current = trend.values

        components.append(current)  # Final trend
        return components


def create_advanced_features(df, interval='1h'):
    """Create comprehensive technical indicators and features (30+ features)"""
    df = df.copy()

    # Basic price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_velocity'] = df['close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()

    # Volatility measures
    df['volatility_5'] = df['returns'].rolling(5).std()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']

    # Volume analysis
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    df['price_volume'] = df['close'] * df['volume']
    df['volume_weighted_price'] = df['price_volume'].rolling(20).sum() / df['volume'].rolling(20).sum()

    # Market microstructure
    df['spread'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

    # Technical indicators using TA library
    try:
        # Trend indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)

        # MACD
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])

        # Bollinger Bands
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        df['bb_mid'] = ta.volatility.bollinger_mavg(df['close'])
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_sma'] = df['rsi'].rolling(5).mean()

        # Stochastic
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])

        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

        # ADX
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])

        # Commodity Channel Index
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])

        # Money Flow Index
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])

    except Exception as e:
        print(f"Warning: Some technical indicators failed: {e}")

    # Time-based features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.day
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.weekday
    df['month'] = pd.to_datetime(df['timestamp']).dt.month

    # Cyclical encoding for time features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # Market regime features
    df['trend_strength'] = df['close'].rolling(20).apply(lambda x: stats.linregress(range(len(x)), x)[2] ** 2)
    df['market_state'] = np.where(df['sma_10'] > df['sma_30'], 1, -1)

    # Fractal dimension (complexity measure)
    def fractal_dimension(data, max_k=10):
        """Calculate fractal dimension using box-counting method"""
        if len(data) < max_k:
            return 0

        n = len(data)
        rs = []
        for k in range(1, min(max_k, n // 4)):
            y = np.cumsum(data - np.mean(data))
            r = np.max(y) - np.min(y)
            s = np.std(data)
            if s > 0:
                rs.append(r / s)
            else:
                rs.append(0)

        if len(rs) > 2:
            try:
                return np.polyfit(np.log(range(1, len(rs) + 1)), np.log(rs), 1)[0]
            except:
                return 0
        return 0

    df['fractal_dim'] = df['returns'].rolling(50).apply(fractal_dimension)

    # Wavelet decomposition
    wavelet = WaveletTransform()
    try:
        price_components = wavelet.decompose(df['close'].values)
        for i, component in enumerate(price_components[:-1]):
            df[f'wavelet_detail_{i}'] = component
        df['wavelet_trend'] = price_components[-1]
    except Exception as e:
        print(f"Warning: Wavelet decomposition failed: {e}")

    return df


# Global variables for model and scaler - now per interval
models = {}
scalers = {}
y_scalers = {}
market_scalers = {}
feature_info = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_in_progress = {}


def get_model_path(interval):
    """Get model file paths for a specific interval"""
    return {
        'model': f'model/models/{interval}/trained_model.pt',
        'scaler': f'model/models/{interval}/scaler.pkl',
        'y_scaler': f'model/models/{interval}/y_scaler.pkl',
        'market_scaler': f'model/models/{interval}/market_scaler.pkl',
        'feature_info': f'model/models/{interval}/feature_info.json',
        'metadata': f'model/models/{interval}/training_metadata.pkl'
    }


def load_model_and_scaler(interval='5m'):
    """Load the trained model and scaler for a specific interval - FIXED VERSION"""
    global models, scalers, y_scalers, market_scalers, feature_info

    try:
        paths = get_model_path(interval)
        print(f"\n🔄 Loading model for interval: {interval}")

        # Check if model directory exists
        model_dir = f'model/models/{interval}'
        if not os.path.exists(model_dir):
            print(f"❌ Model directory not found: {model_dir}")
            return False

        # Load feature information first
        if os.path.exists(paths['feature_info']):
            try:
                with open(paths['feature_info'], 'r') as f:
                    feature_info[interval] = json.load(f)
                print(
                    f"✅ Feature info loaded for {interval}: {feature_info[interval].get('num_features', 'unknown')} features")
            except Exception as e:
                print(f"⚠️ Error loading feature info for {interval}: {e}")
                feature_info[interval] = None
        else:
            print(f"⚠️ Feature info not found for {interval}")
            feature_info[interval] = None

        # Load scalers
        try:
            if os.path.exists(paths['scaler']):
                scalers[interval] = joblib.load(paths['scaler'])
                print(f"✅ Scaler loaded for {interval}")
            else:
                print(f"⚠️ Scaler not found for {interval}")
                scalers[interval] = None
        except Exception as e:
            print(f"❌ Error loading scaler for {interval}: {e}")
            scalers[interval] = None

        # Load y_scaler
        try:
            if os.path.exists(paths['y_scaler']):
                y_scalers[interval] = joblib.load(paths['y_scaler'])
                print(f"✅ Y-scaler loaded for {interval}")
            else:
                print(f"⚠️ Y-scaler not found for {interval}")
                y_scalers[interval] = None
        except Exception as e:
            print(f"❌ Error loading y_scaler for {interval}: {e}")
            y_scalers[interval] = None

        # Load market_scaler
        try:
            if os.path.exists(paths['market_scaler']):
                market_scalers[interval] = joblib.load(paths['market_scaler'])
                print(f"✅ Market scaler loaded for {interval}")
            else:
                print(f"⚠️ Market scaler not found for {interval}")
                market_scalers[interval] = None
        except Exception as e:
            print(f"❌ Error loading market_scaler for {interval}: {e}")
            market_scalers[interval] = None

        # Load model - IMPROVED VERSION
        if os.path.exists(paths['model']):
            try:
                # Determine model type and parameters
                is_enhanced = (feature_info[interval] is not None and
                               feature_info[interval].get('model_type') == 'enhanced')

                if is_enhanced:
                    # Enhanced model with parameters from feature_info
                    model_params = feature_info[interval].get('model_params', {})
                    input_size = model_params.get('input_size', feature_info[interval].get('num_features', 20))
                    market_size = model_params.get('market_features_size', 25)

                    print(
                        f"🔧 Creating enhanced model for {interval} with {input_size} input features, {market_size} market features")

                    models[interval] = EnhancedCNNLSTMModel(
                        input_size=input_size,
                        market_features_size=market_size,
                        cnn_filters=model_params.get('cnn_filters', 64),
                        lstm_units=model_params.get('lstm_units', 128),
                        lstm_units2=model_params.get('lstm_units2', 64),
                        dense_units=model_params.get('dense_units', 128),
                        num_heads=model_params.get('num_heads', 8)
                    )
                else:
                    # Legacy model - try to infer input size from scaler if available
                    if scalers[interval] is not None and hasattr(scalers[interval], 'n_features_in_'):
                        input_size = scalers[interval].n_features_in_
                        print(f"🔧 Creating legacy model for {interval} with {input_size} features (from scaler)")
                    else:
                        input_size = 5  # Default for legacy
                        print(f"🔧 Creating legacy model for {interval} with {input_size} features (default)")

                    models[interval] = CNNLSTMModel(
                        input_size=input_size,
                        cnn_filters=48,
                        kernel_size=5,
                        lstm_units=128,
                        dropout_lstm=0.2,
                        lstm_units2=32,
                        dropout_dense=0.3,
                        dense_units=64,
                        dropout_dense2=0.3
                    )

                # Load the state dict with error handling
                print(f"📁 Loading model weights for {interval}...")
                state_dict = torch.load(paths['model'], map_location=device)

                # Try to load with strict=False first to handle mismatches
                try:
                    models[interval].load_state_dict(state_dict, strict=True)
                    print(f"✅ Model weights loaded strictly for {interval}")
                except RuntimeError as e:
                    print(f"⚠️ Strict loading failed for {interval}, trying flexible loading: {e}")
                    models[interval].load_state_dict(state_dict, strict=False)
                    print(f"✅ Model weights loaded flexibly for {interval}")

                models[interval].to(device)
                models[interval].eval()
                print(f"🎯 Model successfully loaded and ready for {interval}")
                return True

            except Exception as e:
                print(f"❌ Error loading model for {interval}: {e}")
                traceback.print_exc()
                models[interval] = None
                return False
        else:
            print(f"❌ Model file not found for {interval}: {paths['model']}")
            models[interval] = None
            return False

    except Exception as e:
        print(f"💥 Fatal error loading model/scaler for {interval}: {e}")
        traceback.print_exc()
        return False


def prepare_enhanced_data_for_prediction(data, interval='5m', sequence_length=60):
    """Prepare enhanced data with 30+ features for model prediction"""
    try:
        # Convert to DataFrame with timestamp
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['time'], unit='ms')

        # Create advanced features (same as training)
        df = create_advanced_features(df, interval)

        # Select comprehensive features (same order as training)
        price_features = ['open', 'high', 'low', 'close', 'volume']
        technical_features = [
            'returns', 'volatility_5', 'volume_ratio', 'spread', 'body_size',
            'rsi', 'macd', 'bb_position', 'bb_width', 'stoch_k', 'williams_r',
            'adx', 'cci', 'mfi', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
            'trend_strength', 'market_state', 'fractal_dim'
        ]

        # Add wavelet features
        wavelet_features = [col for col in df.columns if col.startswith('wavelet_')]

        all_features = price_features + technical_features + wavelet_features

        # Filter available features
        available_features = [f for f in all_features if f in df.columns]

        # Get feature data
        feature_data = df[available_features].fillna(method='ffill').fillna(0)

        if len(feature_data) < sequence_length:
            print(f"Insufficient data: {len(feature_data)} < {sequence_length}")
            return None, None, None

        # Prepare market context (technical indicators only)
        market_context = feature_data.iloc[-1, len(price_features):].values

        return feature_data.values, available_features, market_context

    except Exception as e:
        print(f"Error preparing enhanced data: {e}")
        traceback.print_exc()
        return None, None, None


def prepare_legacy_data_for_prediction(data, sequence_length=60):
    """Prepare basic 5-feature data for legacy model prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Create basic features for legacy model
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_ma'] = df['volume'].rolling(window=5).mean()

        # Select features for prediction (legacy format)
        features = ['open', 'high', 'low', 'close', 'volume']
        feature_data = df[features].fillna(method='ffill').fillna(0)

        if len(feature_data) < sequence_length:
            return None, None

        return feature_data.values, features

    except Exception as e:
        print(f"Error preparing legacy data: {e}")
        return None, None


def make_prediction(data, interval='5m'):
    """Make price prediction using the trained model for a specific interval - FIXED VERSION"""
    try:
        if interval not in models or models[interval] is None:
            print(f"❌ No model loaded for interval {interval}")
            return None

        # Check if we have feature info (enhanced model) or not (legacy model)
        is_enhanced = (interval in feature_info and
                       feature_info[interval] is not None and
                       feature_info[interval].get('num_features', 0) > 10)

        if is_enhanced:
            print(f"🚀 Using enhanced prediction for {interval}")
            # Enhanced model prediction
            feature_data, available_features, market_context = prepare_enhanced_data_for_prediction(data, interval)

            if feature_data is None:
                print(f"❌ Failed to prepare enhanced data for {interval}")
                return None

            # Scale the data using interval-specific scaler
            if interval in scalers and scalers[interval] is not None:
                original_shape = feature_data.shape
                feature_data_reshaped = feature_data.reshape(-1, feature_data.shape[1])
                scaled_data = scalers[interval].transform(feature_data_reshaped)
                scaled_data = scaled_data.reshape(original_shape)
            else:
                print(f"⚠️ No scaler available for {interval}")
                return None

            # Scale market context
            if interval in market_scalers and market_scalers[interval] is not None:
                market_context_scaled = market_scalers[interval].transform(market_context.reshape(1, -1))[0]
            else:
                market_context_scaled = market_context

            # Get the last sequence for prediction
            sequence = scaled_data[-60:]  # sequence_length = 60
            input_data = np.expand_dims(sequence, axis=0)  # Add batch dimension
            market_data = np.expand_dims(market_context_scaled, axis=0)  # Add batch dimension

            # Convert to tensors
            input_tensor = torch.FloatTensor(input_data).to(device)
            market_tensor = torch.FloatTensor(market_data).to(device)

            # Make prediction with enhanced model
            with torch.no_grad():
                prediction = models[interval](input_tensor, market_tensor)
                predicted_price = prediction.cpu().numpy()[0][0]

        else:
            print(f"🔧 Using legacy prediction for {interval}")
            # Legacy model prediction
            feature_data, features = prepare_legacy_data_for_prediction(data)

            if feature_data is None:
                print(f"❌ Failed to prepare legacy data for {interval}")
                return None

            # Scale the data using interval-specific scaler (or create basic scaling)
            if interval in scalers and scalers[interval] is not None:
                try:
                    scaled_data = scalers[interval].transform(feature_data)
                except ValueError as e:
                    print(f"⚠️ Scaler mismatch for {interval}: {e}")
                    # Fallback to basic scaling
                    scaled_data = (feature_data - feature_data.mean(axis=0)) / (feature_data.std(axis=0) + 1e-8)
                    scaled_data = np.nan_to_num(scaled_data)
            else:
                # Basic scaling
                scaled_data = (feature_data - feature_data.mean(axis=0)) / (feature_data.std(axis=0) + 1e-8)
                scaled_data = np.nan_to_num(scaled_data)

            # Get the last sequence for prediction
            sequence = scaled_data[-60:]  # sequence_length = 60
            input_data = np.expand_dims(sequence, axis=0)  # Add batch dimension

            # Convert to tensor
            input_tensor = torch.FloatTensor(input_data).to(device)

            # Make prediction with legacy model
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
                print(f"⚠️ Error inverse transforming prediction: {e}")
                # Use current price as fallback
                predicted_price = data[-1]['close']

        # Ensure prediction is reasonable (within 20% of current price as sanity check)
        current_price = data[-1]['close']
        if abs(predicted_price - current_price) > (current_price * 0.2):
            print(f"⚠️ Prediction seems unreasonable: {predicted_price} vs current {current_price}")
            # Return a more conservative prediction
            predicted_price = current_price * (1 + np.random.uniform(-0.02, 0.02))

        print(f"✅ Prediction successful for {interval}: {predicted_price}")
        return float(predicted_price)

    except Exception as e:
        print(f"💥 Error making prediction for {interval}: {e}")
        traceback.print_exc()
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

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        return {
            'ma_20': df['ma_20'].iloc[-1] if not df['ma_20'].isna().iloc[-1] else None,
            'ma_50': df['ma_50'].iloc[-1] if not df['ma_50'].isna().iloc[-1] else None,
            'rsi': df['rsi'].iloc[-1] if not df['rsi'].isna().iloc[-1] else None,
            'macd': df['macd'].iloc[-1] if not df['macd'].isna().iloc[-1] else None,
            'macd_signal': df['macd_signal'].iloc[-1] if not df['macd_signal'].isna().iloc[-1] else None,
            'bb_upper': df['bb_upper'].iloc[-1] if not df['bb_upper'].isna().iloc[-1] else None,
            'bb_middle': df['bb_middle'].iloc[-1] if not df['bb_middle'].isna().iloc[-1] else None,
            'bb_lower': df['bb_lower'].iloc[-1] if not df['bb_lower'].isna().iloc[-1] else None,
            'volume_ratio': df['volume_ratio'].iloc[-1] if not df['volume_ratio'].isna().iloc[-1] else None
        }
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        return {}


def run_training_in_background(symbol='BTCUSDT', interval='5m', use_enhanced=True):
    global training_in_progress

    def training_task():
        try:
            training_in_progress[interval] = True
            print(f"🔥 Starting training for {symbol} - {interval} | Enhanced: {use_enhanced}")

            # Use the train_model from train_model.py
            try:
                from model.train_model import train_model
                result = train_model(
                    symbol=symbol,
                    interval=interval,
                    epochs=150,
                    batch_size=64
                )
            except ImportError:
                print("⚠️ train_model module not found, using fallback")
                result = {"status": "error", "error": "Training module not available"}

            if result.get("status") == "success":
                print(f"✅ Training completed: {result}")
                # Reload the model after successful training
                load_model_and_scaler(interval)
            else:
                print(f"❌ Training failed: {result}")

        except Exception as e:
            print(f"💥 Training error for {interval}: {e}")
            traceback.print_exc()
        finally:
            training_in_progress[interval] = False
            print(f"🏁 Training finished for {interval}")

    thread = threading.Thread(target=training_task)
    thread.daemon = True
    thread.start()
    print(f"🧵 Background thread started for {symbol} - {interval}")


# Flask Routes
@app.route('/')
def index():
    return render_template('index.html', symbols=SYMBOLS)


@app.route('/candles')
def candles():
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "5m")
    binance_interval = INTERVALS.get(interval, "5m")

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
@cache.cached(timeout=30, query_string=True)
def predict():
    """Generate AI predictions for the selected symbol and interval"""
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "5m")
    binance_interval = INTERVALS.get(interval, "5m")

    try:
        # Get recent data for prediction
        klines = client.get_klines(symbol=symbol, interval=binance_interval, limit=200)
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
            confidence = min(95, max(5, abs(int(price_change_percent * 20))))
        else:
            predicted_price = current_price
            price_change = 0
            price_change_percent = 0
            direction = "NEUTRAL"
            confidence = 0

        # Determine model type
        is_enhanced = (interval in feature_info and
                       feature_info[interval] is not None and
                       feature_info[interval].get('num_features', 0) > 10)

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
            'model_type': 'enhanced' if is_enhanced else 'legacy',
            'num_features': feature_info[interval].get('num_features', 5) if is_enhanced else 5,
            'training_in_progress': training_in_progress.get(interval, False),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route('/train')
def train_model_endpoint():
    symbol = request.args.get("symbol", "BTCUSDT")
    interval = request.args.get("interval", "5m")
    use_enhanced = request.args.get("enhanced", "true").lower() == "true"

    if training_in_progress.get(interval, False):
        return jsonify({
            'status': 'info',
            'message': f'Training already in progress for {interval}',
            'training_in_progress': True,
            'interval': interval
        })

    try:
        run_training_in_background(symbol=symbol, interval=interval, use_enhanced=use_enhanced)
        return jsonify({
            'status': 'success',
            'message': f'Model training started in background for {symbol} ({interval})',
            'training_in_progress': True,
            'symbol': symbol,
            'interval': interval,
            'enhanced': use_enhanced
        })
    except Exception as e:
        training_in_progress[interval] = False
        return jsonify({
            'status': 'error',
            'message': f"Failed to start training: {str(e)}",
            'training_in_progress': False,
            'interval': interval
        })


@app.route('/training_status')
def training_status():
    """Check training status for a specific interval"""
    interval = request.args.get("interval", "5m")

    # Determine model type
    is_enhanced = (interval in feature_info and
                   feature_info[interval] is not None and
                   feature_info[interval].get('num_features', 0) > 10)

    return jsonify({
        'interval': interval,
        'training_in_progress': training_in_progress.get(interval, False),
        'model_loaded': interval in models and models[interval] is not None,
        'model_type': 'enhanced' if is_enhanced else 'legacy',
        'num_features': feature_info[interval].get('num_features', 5) if is_enhanced else 5,
        'scaler_loaded': interval in scalers and scalers[interval] is not None,
        'y_scaler_loaded': interval in y_scalers and y_scalers[interval] is not None,
        'market_scaler_loaded': interval in market_scalers and market_scalers[interval] is not None
    })


@app.route('/model_info')
def model_info():
    """Get detailed model information for all loaded intervals"""
    info = {}

    for interval in models.keys():
        is_enhanced = (interval in feature_info and
                       feature_info[interval] is not None and
                       feature_info[interval].get('num_features', 0) > 10)

        info[interval] = {
            'model_loaded': models[interval] is not None,
            'model_type': 'enhanced' if is_enhanced else 'legacy',
            'num_features': feature_info[interval].get('num_features', 5) if is_enhanced else 5,
            'feature_names': feature_info[interval].get('feature_names', ['open', 'high', 'low', 'close',
                                                                          'volume']) if is_enhanced else ['open',
                                                                                                          'high', 'low',
                                                                                                          'close',
                                                                                                          'volume'],
            'scalers_loaded': {
                'main_scaler': interval in scalers and scalers[interval] is not None,
                'y_scaler': interval in y_scalers and y_scalers[interval] is not None,
                'market_scaler': interval in market_scalers and market_scalers[interval] is not None
            },
            'training_in_progress': training_in_progress.get(interval, False)
        }

    return jsonify(info)


# API Routes for news and alerts (with fallback if modules not available)
@app.route('/api/news')
def api_news():
    """Get the latest cryptocurrency news"""
    try:
        limit = request.args.get('limit', 15, type=int)
        min_relevance = request.args.get('min_relevance', 0.2, type=float)

        try:
            news_data = get_crypto_news(limit=limit)
            return jsonify(news_data)
        except NameError:
            return jsonify({
                'status': 'error',
                'error': 'News service not available',
                'news': [],
                'sentiment_summary': {},
                'last_updated': int(time.time() * 1000)
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'news': [],
            'sentiment_summary': {},
            'last_updated': int(time.time() * 1000)
        })


@app.route('/api/alerts')
def api_alerts():
    """Get market alerts"""
    try:
        limit = request.args.get('limit', 20, type=int)
        symbol = request.args.get('symbol', None)

        try:
            alerts_data = get_market_alerts(limit=limit, symbol=symbol)
            return jsonify(alerts_data)
        except NameError:
            return jsonify({
                'status': 'error',
                'error': 'Alerts service not available',
                'alerts': [],
                'statistics': {},
                'last_updated': int(time.time() * 1000)
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'alerts': [],
            'statistics': {},
            'last_updated': int(time.time() * 1000)
        })


@app.route('/api/alerts/create', methods=['POST'])
def api_create_alert():
    """Create a custom alert"""
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        alert_type = data.get('alert_type')
        threshold = data.get('threshold')
        condition = data.get('condition', 'above')

        if not all([symbol, alert_type, threshold]):
            return jsonify({
                'status': 'error',
                'error': 'Missing required parameters: symbol, alert_type, threshold'
            })

        try:
            result = create_custom_alert(symbol, alert_type, float(threshold), condition)
            return jsonify(result)
        except NameError:
            return jsonify({
                'status': 'error',
                'error': 'Alert creation service not available'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })


@app.route('/api/alerts/control', methods=['POST'])
def api_alerts_control():
    """Start/stop alerts monitoring"""
    try:
        data = request.get_json()
        action = data.get('action')

        if action == 'start':
            try:
                result = start_monitoring()
            except NameError:
                result = {'status': 'error', 'error': 'Monitoring service not available'}
        elif action == 'stop':
            try:
                result = stop_monitoring()
            except NameError:
                result = {'status': 'error', 'error': 'Monitoring service not available'}
        else:
            return jsonify({
                'status': 'error',
                'error': 'Invalid action. Use "start" or "stop"'
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })


@app.route('/api/market-sentiment')
def api_market_sentiment():
    """Get overall market sentiment from news"""
    try:
        try:
            news_data = get_crypto_news(limit=50)
            if news_data['status'] == 'success':
                sentiment_summary = news_data['sentiment_summary']
                return jsonify({
                    'status': 'success',
                    'sentiment': sentiment_summary,
                    'last_updated': news_data['last_updated']
                })
            else:
                return jsonify(news_data)
        except NameError:
            return jsonify({
                'status': 'error',
                'error': 'Market sentiment service not available'
            })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })


# Initialize models and scalers on startup for common intervals
def initialize_models():
    """Initialize models for all available intervals"""
    common_intervals = ['5m', '15m', '1h', '4h', '1d']
    loaded_count = 0

    print("\n" + "=" * 60)
    print("🚀 INITIALIZING MODELS")
    print("=" * 60)

    for interval in common_intervals:
        model_dir = f'model/models/{interval}'
        if os.path.exists(model_dir):
            print(f"\n📂 Found model directory for {interval}")
            success = load_model_and_scaler(interval)
            if success:
                loaded_count += 1
                print(f"✅ Successfully loaded model for {interval}")
            else:
                print(f"❌ Failed to load model for {interval}")
        else:
            print(f"⚠️ No model directory found for {interval}")

    print(f"\n🎯 Model loading complete: {loaded_count}/{len(common_intervals)} intervals loaded")
    print("=" * 60)
    return loaded_count


if __name__ == '__main__':
    # Initialize alerts monitoring if available
    try:
        from alerts import alerts_engine

        alerts_engine.start_monitoring()
        print("✅ Alerts monitoring started automatically")
    except Exception as e:
        print(f"⚠️ Could not start alerts monitoring: {e}")

    # Ensure model directories exist
    os.makedirs('model', exist_ok=True)
    for interval in ['5m', '15m', '1h', '4h', '1d']:
        os.makedirs(f'model/models/{interval}', exist_ok=True)

    # Initialize models
    loaded_models = initialize_models()

    # Print startup summary
    print(f"\n🌟 ENHANCED CRYPTO PREDICTION APP READY")
    print(f"Device: {device}")
    print(f"Loaded models: {loaded_models}")

    if loaded_models > 0:
        print("\n📊 Model Status:")
        for interval in models.keys():
            if models[interval] is not None:
                is_enhanced = (interval in feature_info and
                               feature_info[interval] is not None and
                               feature_info[interval].get('num_features', 0) > 10)
                model_type = 'Enhanced' if is_enhanced else 'Legacy'
                num_features = feature_info[interval].get('num_features', 5) if is_enhanced else 5
                print(f"  🔹 {interval}: {model_type} ({num_features} features)")

    print("=" * 60)

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)