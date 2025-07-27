# models/train_all_timeframes.py
import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn
import torch.optim as optim
from binance.client import Client
from scipy import stats
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')


class EnhancedCNNLSTMModel(nn.Module):
    def __init__(self, input_size=20, market_features_size=25, cnn_filters=64, kernel_size=3,
                 lstm_units=128, lstm_units2=64, dense_units=128,
                 dropout_lstm=0.3, dropout_dense=0.4, num_heads=8):
        super(EnhancedCNNLSTMModel, self).__init__()

        # Multi-scale CNN layers - Fixed channel distribution
        self.filters_1 = cnn_filters // 3
        self.filters_2 = cnn_filters // 3
        self.filters_3 = cnn_filters - (self.filters_1 + self.filters_2)  # Ensures exact total

        self.conv1_3 = nn.Conv1d(input_size, self.filters_1, 3, padding=1)
        self.conv1_5 = nn.Conv1d(input_size, self.filters_2, 5, padding=2)
        self.conv1_7 = nn.Conv1d(input_size, self.filters_3, 7, padding=3)

        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters * 2, kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv1d(cnn_filters * 2, cnn_filters * 2, kernel_size, padding=kernel_size // 2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)
        self.bn3 = nn.BatchNorm1d(cnn_filters * 2)

        self.pool = nn.AdaptiveMaxPool1d(30)  # Adaptive pooling
        self.dropout_cnn = nn.Dropout(0.25)

        # Attention mechanism - ensure embed_dim is divisible by num_heads
        embed_dim = cnn_filters * 2
        # Adjust num_heads if embed_dim is not divisible
        if embed_dim % num_heads != 0:
            # Find the largest divisor of embed_dim that's <= num_heads
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

        # Feature fusion - dynamically sized based on actual inputs
        feature_size = lstm_units2 * 2  # Bidirectional
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
        self.swish = nn.SiLU()  # Swish activation

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
        x = x + attn_out  # Residual connection

        # Bidirectional LSTM
        x, _ = self.lstm1(x)
        x = self.dropout_dense(x)
        x, _ = self.lstm2(x)
        x = self.dropout_dense(x)

        # Global features
        lstm_out = x[:, -1, :]  # Last output

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
        x = x + residual  # Residual connection

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
    """Create comprehensive technical indicators and features"""
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
    df['market_state'] = np.where(df['sma_10'] > df['sma_30'], 1, -1)  # Bull/Bear

    # Fractal dimension (complexity measure)
    def fractal_dimension(data, max_k=10):
        """Calculate fractal dimension using box-counting method"""
        if len(data) < max_k:
            return 0

        n = len(data)
        rs = []
        for k in range(1, min(max_k, n // 4)):
            # Rescaled range
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
    price_components = wavelet.decompose(df['close'].values)
    for i, component in enumerate(price_components[:-1]):  # Exclude final trend
        df[f'wavelet_detail_{i}'] = component
    df['wavelet_trend'] = price_components[-1]

    return df


def get_market_regime_features(df):
    """Extract market regime features"""
    latest = df.iloc[-1]

    # Volatility regime
    vol_percentile = df['volatility_20'].rank(pct=True).iloc[-1]
    vol_regime = 'high' if vol_percentile > 0.8 else ('low' if vol_percentile < 0.2 else 'medium')

    # Trend regime
    trend_strength = latest.get('trend_strength', 0)
    trend_regime = 'strong' if trend_strength > 0.5 else ('weak' if trend_strength < 0.1 else 'medium')

    # Market state
    market_state = latest.get('market_state', 0)

    return {
        'vol_regime': vol_regime,
        'trend_regime': trend_regime,
        'market_state': market_state,
        'vol_percentile': vol_percentile,
        'trend_strength': trend_strength
    }


def prepare_enhanced_sequences(data, sequence_length=60, prediction_horizon=1, include_future_features=True):
    """Prepare sequences with enhanced features and multiple prediction horizons"""

    # Select comprehensive features
    price_features = ['open', 'high', 'low', 'close', 'volume']
    technical_features = [
        'returns', 'volatility_5', 'volume_ratio', 'spread', 'body_size',
        'rsi', 'macd', 'bb_position', 'bb_width', 'stoch_k', 'williams_r',
        'adx', 'cci', 'mfi', 'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
        'trend_strength', 'market_state', 'fractal_dim'
    ]

    # Add wavelet features
    wavelet_features = [col for col in data.columns if col.startswith('wavelet_')]

    all_features = price_features + technical_features + wavelet_features

    # Filter available features
    available_features = [f for f in all_features if f in data.columns]

    features = data[available_features].fillna(method='ffill').fillna(0).values

    X, y, market_contexts = [], [], []

    for i in range(sequence_length, len(features) - prediction_horizon + 1):
        # Input sequence
        sequence = features[i - sequence_length:i]
        X.append(sequence)

        # Target (future close price)
        target = features[i + prediction_horizon - 1, 3]  # close price
        y.append(target)

        # Market context features (latest values)
        context = features[i - 1, len(price_features):]  # Technical indicators only
        market_contexts.append(context)

    return np.array(X), np.array(y), np.array(market_contexts), available_features


def create_ensemble_model(input_size, market_features_size, device):
    """Create an ensemble of models with different architectures"""
    models = []

    # Model 1: Enhanced CNN-LSTM (64 filters, 128 embed_dim, divisible by 8)
    model1 = EnhancedCNNLSTMModel(
        input_size=input_size,
        market_features_size=market_features_size,
        cnn_filters=64,
        lstm_units=128,
        lstm_units2=64,
        dense_units=128,
        num_heads=8
    ).to(device)

    # Model 2: Different architecture (48 filters, 96 embed_dim, divisible by 8)
    model2 = EnhancedCNNLSTMModel(
        input_size=input_size,
        market_features_size=market_features_size,
        cnn_filters=48,
        lstm_units=96,
        lstm_units2=48,
        dense_units=96,
        num_heads=8
    ).to(device)

    # Model 3: Larger model (80 filters, 160 embed_dim, divisible by 8)
    model3 = EnhancedCNNLSTMModel(
        input_size=input_size,
        market_features_size=market_features_size,
        cnn_filters=80,  # 80*2=160, divisible by 8
        lstm_units=160,
        lstm_units2=80,
        dense_units=160,
        num_heads=8
    ).to(device)

    return [model1, model2, model3]


def train_enhanced_model(symbol='BTCUSDT', interval='1h', epochs=150, batch_size=64):
    """Enhanced training with advanced features"""

    print(f"🚀 Enhanced Training Started: {symbol} - {interval}")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Fetch more data for better training
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=1500)

    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convert data types
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.dropna().reset_index(drop=True)

    print(f"Data points: {len(df)}")

    # Create advanced features
    print("Creating advanced features...")
    df = create_advanced_features(df, interval)
    df = df.dropna().reset_index(drop=True)

    print(f"Data points after feature engineering: {len(df)}")

    # Prepare enhanced sequences
    print("Preparing enhanced sequences...")
    X, y, market_contexts, feature_names = prepare_enhanced_sequences(df, sequence_length=60)

    print(f"Sequences created: {len(X)}")
    print(f"Features per timestep: {X.shape[2]}")
    print(f"Market context features: {market_contexts.shape[1]}")

    # Advanced scaling
    scaler = RobustScaler()  # More robust to outliers
    y_scaler = RobustScaler()
    market_scaler = RobustScaler()

    # Scale features
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[2])
    X_scaled = scaler.fit_transform(X_reshaped).reshape(original_shape)

    # Scale targets and market contexts
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    market_contexts_scaled = market_scaler.fit_transform(market_contexts)

    # Split data
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    market_train, market_test = market_contexts_scaled[:split_idx], market_contexts_scaled[split_idx:]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    market_train_tensor = torch.FloatTensor(market_train)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    market_test_tensor = torch.FloatTensor(market_test)

    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, market_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create ensemble
    ensemble_models = create_ensemble_model(X.shape[2], market_contexts.shape[1], device)
    optimizers = [optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5) for model in ensemble_models]
    schedulers = [optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2) for opt in optimizers]

    # Loss functions
    mse_loss = nn.MSELoss()
    huber_loss = nn.SmoothL1Loss()

    # Training loop with ensemble
    best_losses = [float('inf')] * len(ensemble_models)

    for epoch in range(epochs):
        # Train ensemble
        total_losses = [0] * len(ensemble_models)

        for i, model in enumerate(ensemble_models):
            model.train()
            epoch_loss = 0

            for batch_X, batch_y, batch_market in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                batch_market = batch_market.to(device)

                optimizers[i].zero_grad()
                outputs = model(batch_X, batch_market)

                # Combined loss
                loss1 = mse_loss(outputs.squeeze(), batch_y)
                loss2 = huber_loss(outputs.squeeze(), batch_y)
                loss = 0.7 * loss1 + 0.3 * loss2

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizers[i].step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            total_losses[i] = avg_loss
            schedulers[i].step()

        # Validation
        if epoch % 20 == 0:
            val_losses = []
            for i, model in enumerate(ensemble_models):
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor.to(device), market_test_tensor.to(device))
                    val_loss = mse_loss(val_outputs.squeeze(), y_test_tensor.to(device))
                    val_losses.append(val_loss.item())

                    if val_loss < best_losses[i]:
                        best_losses[i] = val_loss

            print(
                f'Epoch {epoch}: Train losses: {[f"{l:.6f}" for l in total_losses]}, Val losses: {[f"{l:.6f}" for l in val_losses]}')

    # Save ensemble components
    model_dir = f'model/models/{interval}'
    os.makedirs(model_dir, exist_ok=True)

    # Save scalers and metadata
    joblib.dump(scaler, f'{model_dir}/scaler.pkl')
    joblib.dump(y_scaler, f'{model_dir}/y_scaler.pkl')
    joblib.dump(market_scaler, f'{model_dir}/market_scaler.pkl')

    # Save feature information
    feature_info = {
        'feature_names': feature_names,
        'num_features': len(feature_names),
        'sequence_length': 60,
        'market_features': market_contexts.shape[1]
    }

    with open(f'{model_dir}/feature_info.json', 'w') as f:
        json.dump(feature_info, f)

    # Final evaluation with ensemble
    ensemble_predictions = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_tensor.to(device), market_test_tensor.to(device))
            ensemble_predictions.append(pred.cpu().numpy())

    # Average ensemble predictions
    final_predictions = np.mean(ensemble_predictions, axis=0).flatten()

    # Calculate metrics
    mse = np.mean((final_predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(final_predictions - y_test))

    print(f"🎉 Enhanced Training Completed!")
    print(f"Final Ensemble RMSE: {rmse:.6f}")
    print(f"Final Ensemble MAE: {mae:.6f}")

    return {
        'status': 'success',
        'rmse': rmse,
        'mae': mae,
        'num_features': len(feature_names),
        'ensemble_size': len(ensemble_models)
    }


# Enhanced training for all timeframes
def train_all_enhanced_models():
    timeframes = ['5m', '15m', '1h', '4h', '1d']

    for interval in timeframes:
        print(f"\n{'=' * 60}")
        print(f"Training Enhanced Model for {interval}")
        print(f"{'=' * 60}")

        try:
            result = train_enhanced_model(
                symbol='BTCUSDT',
                interval=interval,
                epochs=150,
                batch_size=64
            )
            print(f"✅ {interval} completed: {result}")
        except Exception as e:
            print(f"❌ {interval} failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    train_all_enhanced_models()