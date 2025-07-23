### Live Binance Candlestick Charts with AI Price Predictions

This Docker image provides a powerful web application that combines real-time cryptocurrency market data with AI-powered price predictions. Built with Python, Flask, and PyTorch, it offers traders and enthusiasts an intuitive interface for monitoring Binance markets and accessing AI-generated trading insights.

**Key Features**:
- 📊 Real-time Binance candlestick charts for 25+ popular trading pairs
- 🤖 CNN-LSTM neural network for price prediction
- ⚙️ On-demand model training with customizable parameters
- 📈 Technical indicators (MA, RSI, MACD) visualization
- 🔄 Auto-refresh and auto-prediction capabilities
- 🧠 Self-learning model that improves with new data

**Usage**:
```bash
docker run -d -p 5000:5000 \
  -v ./model:/app/model \
  --name crypto-ai-app \
  yourusername/lstm-crypto-model:tag
```

**Parameters**:
- `-p 5000:5000` - Exposes the web interface on port 5000
- `-v ./model:/app/model` - Persists trained models across container restarts

**After Installation**:
1. Access the web UI at `http://localhost:5000`
2. Select trading pair and time interval
3. Click "Get AI Prediction" for real-time forecasts
4. Use "Train Model" to improve predictions with fresh data

**Technical Specifications**:
- Python 3.9 with PyTorch and Scikit-learn
- Pre-configured CNN-LSTM model architecture
- Automatic data fetching from Binance API
- Atomic file operations for crash-safe training
- GPU acceleration support (via NVIDIA Container Toolkit)

**Ideal For**:
- Crypto traders seeking AI-assisted decision making
- Developers experimenting with financial AI models
- Educators teaching machine learning in finance
- Hobbyists interested in cryptocurrency analysis

*Note: This is not financial advice. Predictions are experimental and should be verified with other analysis methods.*
