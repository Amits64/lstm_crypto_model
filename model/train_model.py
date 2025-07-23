import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from binance.client import Client
from datetime import datetime, timedelta
import warnings
import shutil  # Added for atomic file operations

warnings.filterwarnings('ignore')


# Helper functions for atomic saves
def safe_save(obj, path):
    """Save an object to a path atomically"""
    tmp_path = path + '.tmp'
    try:
        joblib.dump(obj, tmp_path)
        shutil.move(tmp_path, path)
        print(f"Saved {path} atomically")
    except Exception as e:
        print(f"Error saving {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def safe_save_torch(state_dict, path):
    """Save a torch model state_dict atomically"""
    tmp_path = path + '.tmp'
    try:
        torch.save(state_dict, tmp_path)
        shutil.move(tmp_path, path)
        print(f"Saved torch model {path} atomically")
    except Exception as e:
        print(f"Error saving torch model {path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# CNN-LSTM Model Definition (same as in app.py)
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


def get_interval_string(interval):
    """Convert interval to the correct format for Binance API"""
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH
    }

    # Fallback method if the above doesn't work
    if interval not in interval_map:
        # For newer versions of python-binance, use string directly
        return interval

    try:
        return interval_map[interval]
    except (KeyError, AttributeError):
        # If constants don't exist, return the string directly
        return interval


def fetch_training_data(symbol='BTCUSDT', interval='1h', limit=2000):
    """Fetch historical data from Binance for training"""
    client = Client()

    try:
        # Get the correct interval format
        interval_param = get_interval_string(interval)

        # Get historical klines
        klines = client.get_klines(
            symbol=symbol,
            interval=interval_param,
            limit=limit
        )

        if not klines:
            print("No data received from Binance API")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert to appropriate data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Remove any rows with NaN values from conversion
        df = df.dropna(subset=numeric_columns)

        print(f"Successfully fetched {len(df)} data points")
        return df[['timestamp'] + numeric_columns].reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Trying alternative method...")

        # Alternative method using string interval directly
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,  # Use interval string directly
                limit=limit
            )

            if not klines:
                print("No data received from alternative method")
                return None

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.dropna(subset=numeric_columns)

            print(f"Successfully fetched {len(df)} data points using alternative method")
            return df[['timestamp'] + numeric_columns].reset_index(drop=True)

        except Exception as e2:
            print(f"Alternative method also failed: {e2}")
            return None


def create_technical_features(df):
    """Create technical indicators as features"""
    df = df.copy()  # Avoid modifying original dataframe

    # Price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['open_close_ratio'] = df['open'] / df['close']

    # Moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()

    # Volatility
    df['volatility'] = df['close'].rolling(window=10).std()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

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
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    return df


def prepare_sequences(data, sequence_length=60, prediction_horizon=1):
    """Prepare sequences for training"""
    # Select features for training
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    features = data[feature_columns].values

    # Create sequences
    X, y = [], []

    for i in range(sequence_length, len(features) - prediction_horizon + 1):
        # Input sequence
        X.append(features[i - sequence_length:i])
        # Target (next close price)
        y.append(features[i + prediction_horizon - 1, 3])  # close price index is 3

    return np.array(X), np.array(y)


def train_model(symbol='BTCUSDT', epochs=100, batch_size=32, interval='1h'):
    """Train the CNN-LSTM model with best hyperparameters"""

    # Best hyperparameters from your optimization
    best_params = {
        'cnn_filters': 48,
        'kernel_size': 5,
        'lstm_units': 128,
        'dropout_lstm': 0.2,
        'lstm_units2': 32,
        'dropout_dense': 0.3,
        'dense_units': 64,
        'dropout_dense2': 0.3,
        'lr': 0.005594661103196282
    }

    print("Starting model training...")
    print(f"Training for symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Best hyperparameters: {best_params}")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Fetch training data
    print("Fetching training data...")
    df = fetch_training_data(symbol=symbol, interval=interval, limit=2000)

    if df is None:
        return {"error": "Failed to fetch training data"}

    print(f"Fetched {len(df)} data points")

    # Create technical features
    print("Creating technical features...")
    df = create_technical_features(df)

    # Remove NaN values
    df = df.dropna().reset_index(drop=True)
    print(f"Data points after cleaning: {len(df)}")

    if len(df) < 100:
        return {"error": "Insufficient data for training"}

    # Prepare sequences
    print("Preparing sequences...")
    X, y = prepare_sequences(df, sequence_length=60, prediction_horizon=1)
    print(f"Created {len(X)} sequences")

    if len(X) == 0:
        return {"error": "No sequences created - insufficient data"}

    # Scale the data
    print("Scaling data...")
    scaler = MinMaxScaler()

    # Reshape X for scaling
    original_shape = X.shape
    X_reshaped = X.reshape(-1, X.shape[2])
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(original_shape)

    # Scale y separately
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Save the scalers atomically
    os.makedirs('model', exist_ok=True)
    safe_save(scaler, 'model/scaler.pkl')
    safe_save(y_scaler, 'model/y_scaler.pkl')
    print("Scalers saved atomically")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    print("Initializing model...")
    model = CNNLSTMModel(
        input_size=5,
        cnn_filters=best_params['cnn_filters'],
        kernel_size=best_params['kernel_size'],
        lstm_units=best_params['lstm_units'],
        lstm_units2=best_params['lstm_units2'],
        dense_units=best_params['dense_units'],
        dropout_lstm=best_params['dropout_lstm'],
        dropout_dense=best_params['dropout_dense'],
        dropout_dense2=best_params['dropout_dense2']
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test_tensor.to(device)
            y_test_device = y_test_tensor.to(device)
            val_outputs = model(X_test_device)
            val_loss = criterion(val_outputs.squeeze(), y_test_device).item()

        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model atomically
            safe_save_torch(model.state_dict(), 'model/trained_model.pt')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}')

        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test_device = X_test_tensor.to(device)
        test_outputs = model(X_test_device)
        test_predictions = test_outputs.squeeze().cpu().numpy()

        # Calculate metrics
        mse = np.mean((test_predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_predictions - y_test))

        # Calculate accuracy-like metrics
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - test_predictions) / (y_test + 1e-8))) * 100

        # R-squared
        ss_res = np.sum((y_test - test_predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        print(f"\nTraining completed!")
        print(f"Final Test MSE: {mse:.6f}")
        print(f"Final Test RMSE: {rmse:.6f}")
        print(f"Final Test MAE: {mae:.6f}")
        print(f"Final Test MAPE: {mape:.2f}%")
        print(f"Final Test R²: {r2:.4f}")

    # Save training metadata
    training_metadata = {
        "symbol": symbol,
        "interval": interval,
        "epochs_trained": epoch + 1,
        "sequence_length": 60,
        "prediction_horizon": 1,
        "best_params": best_params,
        "final_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2)
        }
    }

    safe_save(training_metadata, 'model/training_metadata.pkl')

    return {
        "status": "success",
        "symbol": symbol,
        "interval": interval,
        "epochs_trained": epoch + 1,
        "final_train_loss": avg_loss,
        "final_val_loss": val_loss,
        "test_mse": mse,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_mape": mape,
        "test_r2": r2,
        "model_saved": "model/trained_model.pt",
        "scaler_saved": "model/scaler.pkl",
        "metadata_saved": "model/training_metadata.pkl"
    }


if __name__ == "__main__":
    # You can customize these parameters
    result = train_model(
        symbol='BTCUSDT',
        epochs=100,
        batch_size=32,
        interval='1h'  # Try '5m', '15m', '1h', '4h', etc.
    )
    print(f"Training result: {result}")