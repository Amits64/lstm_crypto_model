// Ultra-Enhanced Binance Candlestick Chart Script with 100000x Automation
class CandlestickChartAutomation {
  constructor() {
    this.config = {
      currentSymbol: "BTCUSDT",
      currentInterval: "5m",
      maxRetries: 5,
      wsReconnectDelay: 1000,
      batchSize: 50,
      cacheExpiry: 30000,
      alertThresholds: {
        priceChange: 2, // 2%
        volume: 150, // 150% of average
        volatility: 3 // 3 standard deviations
      }
    };

    this.state = {
      isLoading: false,
      wsConnected: false,
      retryCount: 0,
      lastUpdateTime: null,
      activeSubscriptions: new Set(),
      dataCache: new Map(),
      performanceMetrics: {},
      tradingSignals: [],
      watchlist: [],
      alertHistory: []
    };

    this.websocket = null;
    this.refreshInterval = null;
    this.workers = new Map();
    this.eventBus = new EventTarget();

    this.init();
  }

  // ====================
  // INITIALIZATION
  // ====================

  async init() {
    try {
      await this.setupDOMCache();
      await this.setupWebWorkers();
      await this.setupWebSocket();
      await this.setupEventListeners();
      await this.loadUserPreferences();
      await this.initializeMultiSymbolMonitoring();
      await this.startAutomationEngine();

      console.log('🚀 Ultra-Enhanced Automation System Initialized');
    } catch (error) {
      console.error('❌ Initialization failed:', error);
      this.handleCriticalError(error);
    }
  }

  setupDOMCache() {
    this.elements = {
      symbolSelect: document.getElementById("symbolSelect"),
      intervalSelect: document.getElementById("intervalSelect"),
      chart: document.getElementById("chart"),
      lastUpdate: document.getElementById("lastUpdate"),
      errorMessage: document.getElementById("errorMessage"),
      automationPanel: document.getElementById("automationPanel"),
      watchlistPanel: document.getElementById("watchlistPanel"),
      alertsPanel: document.getElementById("alertsPanel"),
      performancePanel: document.getElementById("performancePanel"),
      tradingSignalsPanel: document.getElementById("tradingSignalsPanel"),
      multiChartContainer: document.getElementById("multiChartContainer")
    };

    // Create missing elements dynamically
    this.createMissingElements();
  }

  createMissingElements() {
    const missingElements = [
      { id: 'automationPanel', class: 'automation-panel' },
      { id: 'watchlistPanel', class: 'watchlist-panel' },
      { id: 'alertsPanel', class: 'alerts-panel' },
      { id: 'performancePanel', class: 'performance-panel' },
      { id: 'tradingSignalsPanel', class: 'trading-signals-panel' },
      { id: 'multiChartContainer', class: 'multi-chart-container' }
    ];

    missingElements.forEach(({ id, class: className }) => {
      if (!document.getElementById(id)) {
        const element = document.createElement('div');
        element.id = id;
        element.className = className;
        document.body.appendChild(element);
        this.elements[id.replace('Panel', '').replace('Container', '')] = element;
      }
    });
  }

  // ====================
  // WEB WORKERS SETUP
  // ====================

  async setupWebWorkers() {
    const workerConfigs = [
      { name: 'dataProcessor', script: this.createDataProcessorWorker() },
      { name: 'technicalAnalysis', script: this.createTechnicalAnalysisWorker() },
      { name: 'alertEngine', script: this.createAlertEngineWorker() },
      { name: 'performanceTracker', script: this.createPerformanceTrackerWorker() },
      { name: 'patternRecognition', script: this.createPatternRecognitionWorker() }
    ];

    for (const config of workerConfigs) {
      try {
        const blob = new Blob([config.script], { type: 'application/javascript' });
        const worker = new Worker(URL.createObjectURL(blob));

        worker.onmessage = (e) => this.handleWorkerMessage(config.name, e.data);
        worker.onerror = (e) => this.handleWorkerError(config.name, e);

        this.workers.set(config.name, worker);
        console.log(`✅ Worker ${config.name} initialized`);
      } catch (error) {
        console.error(`❌ Failed to initialize worker ${config.name}:`, error);
      }
    }
  }

  createDataProcessorWorker() {
    return `
      self.onmessage = function(e) {
        const { type, data, config } = e.data;

        switch(type) {
          case 'PROCESS_CANDLES':
            const processed = processCandles(data, config);
            self.postMessage({ type: 'CANDLES_PROCESSED', data: processed });
            break;
        default:
          audio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMG';
      }

      audio.play().catch(e => console.log('🔇 Audio play failed:', e));
    } catch (error) {
      console.log('🔇 Audio not supported');
    }
  }

  sendBrowserNotification(alert) {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(`${alert.symbol || this.config.currentSymbol} Alert`, {
        body: alert.message,
        icon: '/favicon.ico',
        tag: alert.type
      });
    } else if ('Notification' in window && Notification.permission !== 'denied') {
      Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
          this.sendBrowserNotification(alert);
        }
      });
    }
  }

  // ====================
  // ADVANCED FEATURES
  // ====================

  async startAdvancedPatternRecognition() {
    setInterval(async () => {
      try {
        const cachedData = this.getCachedData();
        if (cachedData && cachedData.length > 50) {
          this.workers.get('patternRecognition')?.postMessage({
            type: 'RECOGNIZE_PATTERNS',
            data: cachedData,
            config: { minConfidence: 0.7 }
          });
        }
      } catch (error) {
        console.error('❌ Pattern recognition error:', error);
      }
    }, 120000); // Every 2 minutes
  }

  startRiskManagement() {
    setInterval(async () => {
      try {
        const riskMetrics = await this.calculateRiskMetrics();
        this.updateRiskDisplay(riskMetrics);

        // Check for risk thresholds
        if (riskMetrics.overallRisk > 0.8) {
          this.triggerRiskAlert(riskMetrics);
        }

      } catch (error) {
        console.error('❌ Risk management error:', error);
      }
    }, 30000); // Every 30 seconds
  }

  async calculateRiskMetrics() {
    const portfolio = await this.calculatePortfolioMetrics();
    const marketData = this.getCachedData();

    if (!marketData || marketData.length < 20) {
      return { overallRisk: 0, volatility: 0, correlation: 0 };
    }

    const returns = [];
    for (let i = 1; i < marketData.length; i++) {
      const ret = (marketData[i].close - marketData[i-1].close) / marketData[i-1].close;
      returns.push(ret);
    }

    const volatility = this.calculateVolatility(returns);
    const var95 = this.calculateVaR(returns, 0.95);
    const maxDrawdown = this.calculateMaxDrawdown(marketData);

    return {
      overallRisk: Math.min(volatility * 10, 1), // Normalize to 0-1
      volatility: volatility,
      var95: var95,
      maxDrawdown: maxDrawdown,
      portfolio: portfolio
    };
  }

  calculateVolatility(returns) {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    return Math.sqrt(variance * 252); // Annualized volatility
  }

  calculateVaR(returns, confidence) {
    const sorted = returns.sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sorted.length);
    return sorted[index];
  }

  calculateMaxDrawdown(data) {
    let maxDrawdown = 0;
    let peak = data[0].close;

    for (let i = 1; i < data.length; i++) {
      if (data[i].close > peak) {
        peak = data[i].close;
      }

      const drawdown = (peak - data[i].close) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown;
  }

  startBacktestingEngine() {
    // Initialize backtesting with historical data
    this.backtestStrategies();
  }

  async backtestStrategies() {
    const strategies = [
      { name: 'RSI_MEAN_REVERSION', params: { rsiPeriod: 14, oversold: 30, overbought: 70 } },
      { name: 'MA_CROSSOVER', params: { fastMA: 10, slowMA: 20 } },
      { name: 'BOLLINGER_BANDS', params: { period: 20, stdDev: 2 } },
      { name: 'MOMENTUM', params: { period: 14, threshold: 0.02 } }
    ];

    for (const strategy of strategies) {
      try {
        const results = await this.runBacktest(strategy);
        this.updateBacktestResults(strategy.name, results);
      } catch (error) {
        console.error(`❌ Backtest error for ${strategy.name}:`, error);
      }
    }
  }

  async runBacktest(strategy) {
    const historicalData = this.getCachedData();
    if (!historicalData || historicalData.length < 100) {
      return { error: 'Insufficient data' };
    }

    const trades = [];
    let position = null;
    let balance = 10000; // Starting balance

    for (let i = 50; i < historicalData.length - 1; i++) {
      const signal = this.evaluateStrategy(strategy, historicalData, i);

      if (signal === 'BUY' && !position) {
        position = {
          type: 'LONG',
          entry: historicalData[i].close,
          timestamp: historicalData[i].time,
          size: balance * 0.95 / historicalData[i].close
        };
      } else if (signal === 'SELL' && position) {
        const exit = historicalData[i].close;
        const profit = (exit - position.entry) * position.size;
        balance += profit;

        trades.push({
          ...position,
          exit: exit,
          exitTime: historicalData[i].time,
          profit: profit,
          return: (exit - position.entry) / position.entry
        });

        position = null;
      }
    }

    return this.calculateBacktestMetrics(trades, balance);
  }

  evaluateStrategy(strategy, data, index) {
    const slice = data.slice(Math.max(0, index - 50), index + 1);

    switch (strategy.name) {
      case 'RSI_MEAN_REVERSION':
        return this.evaluateRSIStrategy(slice, strategy.params);
      case 'MA_CROSSOVER':
        return this.evaluateMAStrategy(slice, strategy.params);
      case 'BOLLINGER_BANDS':
        return this.evaluateBollingerStrategy(slice, strategy.params);
      case 'MOMENTUM':
        return this.evaluateMomentumStrategy(slice, strategy.params);
      default:
        return 'HOLD';
    }
  }

  evaluateRSIStrategy(data, params) {
    if (data.length < params.rsiPeriod + 1) return 'HOLD';

    const closes = data.map(d => d.close);
    const rsi = this.calculateRSI(closes, params.rsiPeriod);
    const currentRSI = rsi[rsi.length - 1];

    if (currentRSI < params.oversold) return 'BUY';
    if (currentRSI > params.overbought) return 'SELL';
    return 'HOLD';
  }

  evaluateMAStrategy(data, params) {
    if (data.length < Math.max(params.fastMA, params.slowMA) + 1) return 'HOLD';

    const closes = data.map(d => d.close);
    const fastMA = this.calculateSMA(closes, params.fastMA);
    const slowMA = this.calculateSMA(closes, params.slowMA);

    const currentFast = fastMA[fastMA.length - 1];
    const currentSlow = slowMA[slowMA.length - 1];
    const prevFast = fastMA[fastMA.length - 2];
    const prevSlow = slowMA[slowMA.length - 2];

    if (currentFast > currentSlow && prevFast <= prevSlow) return 'BUY';
    if (currentFast < currentSlow && prevFast >= prevSlow) return 'SELL';
    return 'HOLD';
  }

  calculateBacktestMetrics(trades, finalBalance) {
    if (trades.length === 0) {
      return { totalReturn: 0, winRate: 0, sharpeRatio: 0, maxDrawdown: 0 };
    }

    const totalReturn = (finalBalance - 10000) / 10000;
    const winningTrades = trades.filter(t => t.profit > 0);
    const winRate = winningTrades.length / trades.length;

    const returns = trades.map(t => t.return);
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const stdReturn = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
    const sharpeRatio = avgReturn / stdReturn * Math.sqrt(252);

    return {
      totalTrades: trades.length,
      winningTrades: winningTrades.length,
      totalReturn: totalReturn,
      winRate: winRate,
      sharpeRatio: sharpeRatio,
      avgReturn: avgReturn,
      maxDrawdown: this.calculateTradeDrawdown(trades)
    };
  }

  // ====================
  // NEWS SENTIMENT ANALYSIS
  // ====================

  startNewsSentimentAnalysis() {
    setInterval(async () => {
      try {
        const news = await this.fetchLatestNews();
        const sentiment = await this.analyzeSentiment(news);
        this.updateSentimentDisplay(sentiment);

        // Generate sentiment-based signals
        const sentimentSignals = this.generateSentimentSignals(sentiment);
        if (sentimentSignals.length > 0) {
          this.handleSentimentSignals(sentimentSignals);
        }

      } catch (error) {
        console.error('❌ News sentiment analysis error:', error);
      }
    }, 300000); // Every 5 minutes
  }

  async fetchLatestNews() {
    try {
      const response = await fetch(`/api/news?_=${Date.now()}`);
      const data = await response.json();

      if (data.status === 'success' && data.news.length > 0) {
        return data.news.map(item => ({
          title: item.title,
          sentiment: item.sentiment,
          source: item.source,
          timestamp: item.timestamp
        }));
      } else {
        return []; // No news
      }
    } catch (error) {
      console.error('❌ Failed to fetch live news:', error);
      return []; // Fallback to empty
    }
  }

  async analyzeSentiment(news) {
    const sentimentScore = news.reduce((sum, article) => sum + article.sentiment, 0) / news.length;

    return {
      overall: sentimentScore,
      bullish: news.filter(n => n.sentiment > 0.3).length,
      bearish: news.filter(n => n.sentiment < -0.3).length,
      neutral: news.filter(n => Math.abs(n.sentiment) <= 0.3).length,
      articles: news
    };
  }

  generateSentimentSignals(sentiment) {
    const signals = [];

    if (sentiment.overall > 0.6) {
      signals.push({
        type: 'SENTIMENT_BULLISH',
        strength: sentiment.overall,
        message: `Strong bullish sentiment detected (${(sentiment.overall * 100).toFixed(1)}%)`,
        signal: 'BUY'
      });
    } else if (sentiment.overall < -0.6) {
      signals.push({
        type: 'SENTIMENT_BEARISH',
        strength: Math.abs(sentiment.overall),
        message: `Strong bearish sentiment detected (${(Math.abs(sentiment.overall) * 100).toFixed(1)}%)`,
        signal: 'SELL'
      });
    }

    return signals;
  }

  // ====================
  // MACHINE LEARNING FEATURES
  // ====================

  async initializeMLModels() {
    try {
      // Initialize TensorFlow.js models
      this.models = {
        pricePredictor: await this.loadPricePredictionModel(),
        patternClassifier: await this.loadPatternClassificationModel(),
        riskAssessment: await this.loadRiskAssessmentModel()
      };

      console.log('🧠 ML Models initialized');
    } catch (error) {
      console.error('❌ ML initialization failed:', error);
    }
  }

  async loadPricePredictionModel() {
    // In real implementation, load pre-trained TensorFlow.js model
    // return await tf.loadLayersModel('/models/price-predictor/model.json');

    // Mock model for demonstration
    return {
      predict: (data) => {
        // Simple mock prediction
        const lastPrice = data[data.length - 1];
        const trend = (data[data.length - 1] - data[data.length - 10]) / data[data.length - 10];
        return lastPrice * (1 + trend * 0.1 + (Math.random() - 0.5) * 0.02);
      }
    };
  }

  async predictNextPrice(timeframe = '1h') {
    try {
      const data = this.getCachedData();
      if (!data || data.length < 50) return null;

      const prices = data.map(d => d.close);
      const features = this.extractFeatures(data);

      const prediction = this.models.pricePredictor.predict(prices);

      return {
        predicted: prediction,
        confidence: this.calculatePredictionConfidence(data),
        timeframe: timeframe,
        timestamp: Date.now()
      };

    } catch (error) {
      console.error('❌ Price prediction error:', error);
      return null;
    }
  }

  extractFeatures(data) {
    const features = [];

    for (let i = 20; i < data.length; i++) {
      const slice = data.slice(i - 20, i);
      const prices = slice.map(d => d.close);
      const volumes = slice.map(d => d.volume);

      features.push({
        sma5: this.calculateSMA(prices, 5)[0],
        sma10: this.calculateSMA(prices, 10)[0],
        sma20: this.calculateSMA(prices, 20)[0],
        rsi: this.calculateRSI(prices, 14)[0],
        volume: volumes[volumes.length - 1],
        volumeMA: volumes.reduce((sum, v) => sum + v, 0) / volumes.length,
        priceChange: (prices[prices.length - 1] - prices[0]) / prices[0],
        volatility: this.calculateVolatility(prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]))
      });
    }

    return features;
  }

  // ====================
  // UTILITY FUNCTIONS
  // ====================

  calculateSMA(values, period) {
    const result = [];
    for (let i = period - 1; i < values.length; i++) {
      const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
    return result;
  }

  calculateRSI(values, period) {
    const gains = [];
    const losses = [];

    for (let i = 1; i < values.length; i++) {
      const change = values[i] - values[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }

    const result = [];
    for (let i = period - 1; i < gains.length; i++) {
      const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
      const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
      const rs = avgGain / avgLoss;
      result.push(100 - (100 / (1 + rs)));
    }
    return result;
  }

  getCachedData(symbol = null, interval = null) {
    const key = `${symbol || this.config.currentSymbol}_${interval || this.config.currentInterval}`;
    const cached = this.state.dataCache.get(key);
    return cached ? cached.data : null;
  }

  trackPerformance(type, data) {
    this.workers.get('performanceTracker')?.postMessage({
      type: type.toUpperCase(),
      data: data
    });
  }

  // ====================
  // EVENT HANDLERS
  // ====================

  async setupEventListeners() {
    // Symbol and interval changes
    this.elements.symbolSelect?.addEventListener("change", (e) => {
      this.config.currentSymbol = e.target.value;
      this.handleSymbolChange();
    });

    this.elements.intervalSelect?.addEventListener("change", (e) => {
      this.config.currentInterval = e.target.value;
      this.handleIntervalChange();
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));

    // Page visibility changes
    document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));

    // Window resize
    window.addEventListener('resize', this.debounce(this.handleResize.bind(this), 250));

    // Custom events
    this.eventBus.addEventListener('alert', this.handleCustomAlert.bind(this));
    this.eventBus.addEventListener('signal', this.handleTradingSignal.bind(this));
  }

  handleSymbolChange() {
    this.resetRefreshInterval();
    this.fetchAndPlot();

    // Update subscriptions
    if (this.state.wsConnected) {
      this.subscribeToSymbol(this.config.currentSymbol, this.config.currentInterval);
    }
  }

  handleIntervalChange() {
    this.resetRefreshInterval();
    this.fetchAndPlot();

    // Update refresh rate
    const refreshRate = this.getRefreshRate(this.config.currentInterval);
    console.log(`🔄 Refresh rate updated to ${refreshRate / 1000}s for ${this.config.currentInterval}`);
  }

  handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + R for manual refresh
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
      e.preventDefault();
      this.fetchAndPlot();
    }

    // Space for pause/resume
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
      e.preventDefault();
      this.toggleAutoRefresh();
    }

    // Number keys for quick interval changes
    const intervalMap = {
      '1': '1m', '2': '5m', '3': '15m', '4': '1h', '5': '4h', '6': '1d'
    };

    if (intervalMap[e.key] && !e.ctrlKey && !e.metaKey) {
      this.config.currentInterval = intervalMap[e.key];
      if (this.elements.intervalSelect) {
        this.elements.intervalSelect.value = this.config.currentInterval;
      }
      this.handleIntervalChange();
    }
  }

  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // ====================
  // CLEANUP AND ERROR HANDLING
  // ====================

  handleCriticalError(error) {
    console.error('💥 Critical Error:', error);

    // Show user-friendly error message
    this.showErrorMessage(`System Error: ${error.message}. Please refresh the page.`);

    // Attempt recovery
    setTimeout(() => {
      this.attemptRecovery();
    }, 5000);
  }

  async attemptRecovery() {
    try {
      console.log('🔄 Attempting system recovery...');

      // Clear all intervals
      clearInterval(this.refreshInterval);

      // Reconnect WebSocket
      if (!this.state.wsConnected) {
        await this.setupWebSocket();
      }

      // Restart data fetching
      this.fetchAndPlot();

      console.log('✅ System recovery successful');
    } catch (error) {
      console.error('❌ Recovery failed:', error);
    }
  }

  cleanup() {
    // Clear intervals
    clearInterval(this.refreshInterval);

    // Close WebSocket
    if (this.websocket) {
      this.websocket.close();
    }

    // Terminate workers
    this.workers.forEach((worker, name) => {
      worker.terminate();
      console.log(`🔌 Worker ${name} terminated`);
    });

    // Clear caches
    this.state.dataCache.clear();

    console.log('🧹 Cleanup completed');
  }

  // ====================
  // MAIN EXECUTION
  // ====================

  async fetchAndPlot() {
    if (this.state.isLoading) {
      console.log("⏳ Request already in progress, skipping...");
      return;
    }

    this.state.isLoading = true;
    this.showLoading();

    try {
      const data = await this.fetchCandleData(this.config.currentSymbol, this.config.currentInterval);

      if (!data || data.length === 0) {
        throw new Error("No data received from server");
      }

      this.state.retryCount = 0;
      await this.plotCandlestickChart(data);
      this.updateLastUpdateTime();

      // Process data through workers
      await this.processCandleData(data[data.length - 1]);

    } catch (error) {
      this.handleError(error, 'fetching and plotting data');
    } finally {
      this.state.isLoading = false;
    }
  }

  async fetchCandleData(symbol, interval) {
    const timestamp = Date.now();
    const url = `/candles?symbol=${symbol}&interval=${interval}&_t=${timestamp}`;

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }

    return data;
  }

  getRefreshRate(interval) {
    const refreshRates = {
      '1m': 5000,    '3m': 10000,   '5m': 15000,   '15m': 30000,
      '30m': 60000,  '1h': 120000,  '2h': 240000,  '4h': 300000,
      '6h': 360000,  '8h': 480000,  '12h': 720000, '1d': 1800000,
      '3d': 3600000, '1w': 7200000, '1M': 14400000
    };
    return refreshRates[interval] || 15000;
  }

  resetRefreshInterval() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }
    const refreshRate = this.getRefreshRate(this.config.currentInterval);
    this.refreshInterval = setInterval(() => this.fetchAndPlot(), refreshRate);
  }

  // Initialize the system
  static async create() {
    const automation = new CandlestickChartAutomation();
    await automation.init();
    return automation;
  }
}

// ====================
// GLOBAL INITIALIZATION
// ====================

// Auto-initialize when DOM is ready
if (typeof window !== 'undefined') {
  let automationSystem = null;

  const initializeSystem = async () => {
    try {
      console.log('🚀 Initializing Ultra-Enhanced Candlestick Automation System...');
      automationSystem = await CandlestickChartAutomation.create();

      // Expose to global scope for debugging
      window.candlestickAutomation = automationSystem;

      console.log('✅ System fully initialized and ready!');
    } catch (error) {
      console.error('💥 System initialization failed:', error);
    }
  };

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSystem);
  } else {
    initializeSystem();
  }

  // Cleanup on page unload
  window.addEventListener('beforeunload', () => {
    if (automationSystem) {
      automationSystem.cleanup();
    }
  });
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
  module.exports = CandlestickChartAutomation;
}

          case 'CALCULATE_INDICATORS':
            const indicators = calculateTechnicalIndicators(data, config);
            self.postMessage({ type: 'INDICATORS_CALCULATED', data: indicators });
            break;

          case 'BATCH_PROCESS':
            const results = data.map(item => processCandles(item.data, item.config));
            self.postMessage({ type: 'BATCH_PROCESSED', data: results });
            break;
        }
      };

      function processCandles(rawData, config) {
        return rawData.map(candle => ({
          ...candle,
          bodySize: Math.abs(candle.close - candle.open),
          upperWick: candle.high - Math.max(candle.open, candle.close),
          lowerWick: Math.min(candle.open, candle.close) - candle.low,
          isGreen: candle.close > candle.open,
          volume24h: candle.volume,
          timestamp: new Date(candle.time).getTime()
        }));
      }

      function calculateTechnicalIndicators(data, config) {
        const closes = data.map(d => d.close);
        const volumes = data.map(d => d.volume);

        return {
          sma: calculateSMA(closes, config.period || 20),
          ema: calculateEMA(closes, config.period || 20),
          rsi: calculateRSI(closes, config.rsiPeriod || 14),
          macd: calculateMACD(closes),
          bollingerBands: calculateBollingerBands(closes, config.period || 20),
          volumeProfile: calculateVolumeProfile(data),
          support: findSupportLevels(data),
          resistance: findResistanceLevels(data)
        };
      }

      function calculateSMA(values, period) {
        const result = [];
        for (let i = period - 1; i < values.length; i++) {
          const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
          result.push(sum / period);
        }
        return result;
      }

      function calculateEMA(values, period) {
        const result = [];
        const multiplier = 2 / (period + 1);
        result[0] = values[0];

        for (let i = 1; i < values.length; i++) {
          result[i] = (values[i] * multiplier) + (result[i - 1] * (1 - multiplier));
        }
        return result;
      }

      function calculateRSI(values, period) {
        const gains = [];
        const losses = [];

        for (let i = 1; i < values.length; i++) {
          const change = values[i] - values[i - 1];
          gains.push(change > 0 ? change : 0);
          losses.push(change < 0 ? Math.abs(change) : 0);
        }

        const result = [];
        for (let i = period - 1; i < gains.length; i++) {
          const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
          const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
          const rs = avgGain / avgLoss;
          result.push(100 - (100 / (1 + rs)));
        }
        return result;
      }

      function calculateMACD(values) {
        const ema12 = calculateEMA(values, 12);
        const ema26 = calculateEMA(values, 26);
        const macdLine = ema12.map((val, i) => val - ema26[i]).filter(val => !isNaN(val));
        const signalLine = calculateEMA(macdLine, 9);
        const histogram = macdLine.map((val, i) => val - (signalLine[i] || 0));

        return { macdLine, signalLine, histogram };
      }

      function calculateBollingerBands(values, period) {
        const sma = calculateSMA(values, period);
        const result = [];

        for (let i = period - 1; i < values.length; i++) {
          const slice = values.slice(i - period + 1, i + 1);
          const mean = sma[i - period + 1];
          const stdDev = Math.sqrt(slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period);

          result.push({
            upper: mean + (2 * stdDev),
            middle: mean,
            lower: mean - (2 * stdDev)
          });
        }
        return result;
      }

      function calculateVolumeProfile(data) {
        const profile = {};
        data.forEach(candle => {
          const priceLevel = Math.round(candle.close * 100) / 100;
          profile[priceLevel] = (profile[priceLevel] || 0) + candle.volume;
        });
        return profile;
      }

      function findSupportLevels(data) {
        const lows = data.map(d => d.low);
        const supports = [];

        for (let i = 2; i < lows.length - 2; i++) {
          if (lows[i] < lows[i-1] && lows[i] < lows[i-2] &&
              lows[i] < lows[i+1] && lows[i] < lows[i+2]) {
            supports.push({ price: lows[i], timestamp: data[i].time, strength: 1 });
          }
        }
        return supports;
      }

      function findResistanceLevels(data) {
        const highs = data.map(d => d.high);
        const resistances = [];

        for (let i = 2; i < highs.length - 2; i++) {
          if (highs[i] > highs[i-1] && highs[i] > highs[i-2] &&
              highs[i] > highs[i+1] && highs[i] > highs[i+2]) {
            resistances.push({ price: highs[i], timestamp: data[i].time, strength: 1 });
          }
        }
        return resistances;
      }
    `;
  }

  createTechnicalAnalysisWorker() {
    return `
      self.onmessage = function(e) {
        const { type, data, config } = e.data;

        switch(type) {
          case 'ANALYZE_PATTERNS':
            const patterns = analyzePatterns(data, config);
            self.postMessage({ type: 'PATTERNS_ANALYZED', data: patterns });
            break;

          case 'GENERATE_SIGNALS':
            const signals = generateTradingSignals(data, config);
            self.postMessage({ type: 'SIGNALS_GENERATED', data: signals });
            break;

          case 'CALCULATE_FIBONACCI':
            const fibonacci = calculateFibonacci(data, config);
            self.postMessage({ type: 'FIBONACCI_CALCULATED', data: fibonacci });
            break;
        }
      };

      function analyzePatterns(data, config) {
        const patterns = [];

        // Doji patterns
        patterns.push(...findDojiPatterns(data));

        // Hammer and Hanging Man
        patterns.push(...findHammerPatterns(data));

        // Engulfing patterns
        patterns.push(...findEngulfingPatterns(data));

        // Head and Shoulders
        patterns.push(...findHeadAndShouldersPatterns(data));

        // Double Top/Bottom
        patterns.push(...findDoubleTopBottomPatterns(data));

        // Triangle patterns
        patterns.push(...findTrianglePatterns(data));

        return patterns;
      }

      function findDojiPatterns(data) {
        const patterns = [];
        data.forEach((candle, index) => {
          const bodySize = Math.abs(candle.close - candle.open);
          const totalRange = candle.high - candle.low;

          if (bodySize / totalRange < 0.1) {
            patterns.push({
              type: 'DOJI',
              timestamp: candle.time,
              index: index,
              confidence: 0.7,
              signal: 'REVERSAL'
            });
          }
        });
        return patterns;
      }

      function findHammerPatterns(data) {
        const patterns = [];
        data.forEach((candle, index) => {
          const bodySize = Math.abs(candle.close - candle.open);
          const lowerWick = Math.min(candle.open, candle.close) - candle.low;
          const upperWick = candle.high - Math.max(candle.open, candle.close);

          if (lowerWick > bodySize * 2 && upperWick < bodySize * 0.5) {
            patterns.push({
              type: 'HAMMER',
              timestamp: candle.time,
              index: index,
              confidence: 0.8,
              signal: 'BULLISH'
            });
          }
        });
        return patterns;
      }

      function findEngulfingPatterns(data) {
        const patterns = [];
        for (let i = 1; i < data.length; i++) {
          const prev = data[i - 1];
          const curr = data[i];

          // Bullish Engulfing
          if (prev.close < prev.open && curr.close > curr.open &&
              curr.open < prev.close && curr.close > prev.open) {
            patterns.push({
              type: 'BULLISH_ENGULFING',
              timestamp: curr.time,
              index: i,
              confidence: 0.85,
              signal: 'BULLISH'
            });
          }

          // Bearish Engulfing
          if (prev.close > prev.open && curr.close < curr.open &&
              curr.open > prev.close && curr.close < prev.open) {
            patterns.push({
              type: 'BEARISH_ENGULFING',
              timestamp: curr.time,
              index: i,
              confidence: 0.85,
              signal: 'BEARISH'
            });
          }
        }
        return patterns;
      }

      function findHeadAndShouldersPatterns(data) {
        // Simplified Head and Shoulders detection
        const patterns = [];
        const peaks = findPeaks(data);

        for (let i = 2; i < peaks.length; i++) {
          const left = peaks[i - 2];
          const head = peaks[i - 1];
          const right = peaks[i];

          if (head.price > left.price && head.price > right.price &&
              Math.abs(left.price - right.price) / left.price < 0.02) {
            patterns.push({
              type: 'HEAD_AND_SHOULDERS',
              timestamp: head.timestamp,
              confidence: 0.9,
              signal: 'BEARISH',
              neckline: (left.price + right.price) / 2
            });
          }
        }
        return patterns;
      }

      function findDoubleTopBottomPatterns(data) {
        const patterns = [];
        const peaks = findPeaks(data);
        const troughs = findTroughs(data);

        // Double Top
        for (let i = 1; i < peaks.length; i++) {
          const first = peaks[i - 1];
          const second = peaks[i];

          if (Math.abs(first.price - second.price) / first.price < 0.02) {
            patterns.push({
              type: 'DOUBLE_TOP',
              timestamp: second.timestamp,
              confidence: 0.8,
              signal: 'BEARISH'
            });
          }
        }

        // Double Bottom
        for (let i = 1; i < troughs.length; i++) {
          const first = troughs[i - 1];
          const second = troughs[i];

          if (Math.abs(first.price - second.price) / first.price < 0.02) {
            patterns.push({
              type: 'DOUBLE_BOTTOM',
              timestamp: second.timestamp,
              confidence: 0.8,
              signal: 'BULLISH'
            });
          }
        }

        return patterns;
      }

      function findTrianglePatterns(data) {
        // Simplified triangle pattern detection
        const patterns = [];
        const window = 20;

        for (let i = window; i < data.length - window; i++) {
          const slice = data.slice(i - window, i + window);
          const highs = slice.map(d => d.high);
          const lows = slice.map(d => d.low);

          const highTrend = calculateTrendSlope(highs);
          const lowTrend = calculateTrendSlope(lows);

          if (highTrend < -0.001 && lowTrend > 0.001) {
            patterns.push({
              type: 'ASCENDING_TRIANGLE',
              timestamp: data[i].time,
              confidence: 0.7,
              signal: 'BULLISH'
            });
          } else if (highTrend > 0.001 && lowTrend < -0.001) {
            patterns.push({
              type: 'DESCENDING_TRIANGLE',
              timestamp: data[i].time,
              confidence: 0.7,
              signal: 'BEARISH'
            });
          }
        }

        return patterns;
      }

      function generateTradingSignals(data, config) {
        const signals = [];

        // RSI signals
        const rsi = calculateRSI(data.map(d => d.close), 14);
        rsi.forEach((value, index) => {
          if (value < 30) {
            signals.push({
              type: 'RSI_OVERSOLD',
              timestamp: data[index + 14]?.time,
              signal: 'BUY',
              strength: (30 - value) / 30,
              price: data[index + 14]?.close
            });
          } else if (value > 70) {
            signals.push({
              type: 'RSI_OVERBOUGHT',
              timestamp: data[index + 14]?.time,
              signal: 'SELL',
              strength: (value - 70) / 30,
              price: data[index + 14]?.close
            });
          }
        });

        // Moving Average Crossovers
        const sma20 = calculateSMA(data.map(d => d.close), 20);
        const sma50 = calculateSMA(data.map(d => d.close), 50);

        for (let i = 1; i < Math.min(sma20.length, sma50.length); i++) {
          if (sma20[i] > sma50[i] && sma20[i - 1] <= sma50[i - 1]) {
            signals.push({
              type: 'MA_GOLDEN_CROSS',
              timestamp: data[i + 50]?.time,
              signal: 'BUY',
              strength: 0.8,
              price: data[i + 50]?.close
            });
          } else if (sma20[i] < sma50[i] && sma20[i - 1] >= sma50[i - 1]) {
            signals.push({
              type: 'MA_DEATH_CROSS',
              timestamp: data[i + 50]?.time,
              signal: 'SELL',
              strength: 0.8,
              price: data[i + 50]?.close
            });
          }
        }

        return signals;
      }

      function calculateFibonacci(data, config) {
        const high = Math.max(...data.map(d => d.high));
        const low = Math.min(...data.map(d => d.low));
        const range = high - low;

        return {
          '0%': high,
          '23.6%': high - (range * 0.236),
          '38.2%': high - (range * 0.382),
          '50%': high - (range * 0.5),
          '61.8%': high - (range * 0.618),
          '78.6%': high - (range * 0.786),
          '100%': low
        };
      }

      function findPeaks(data) {
        const peaks = [];
        for (let i = 1; i < data.length - 1; i++) {
          if (data[i].high > data[i - 1].high && data[i].high > data[i + 1].high) {
            peaks.push({ price: data[i].high, timestamp: data[i].time, index: i });
          }
        }
        return peaks;
      }

      function findTroughs(data) {
        const troughs = [];
        for (let i = 1; i < data.length - 1; i++) {
          if (data[i].low < data[i - 1].low && data[i].low < data[i + 1].low) {
            troughs.push({ price: data[i].low, timestamp: data[i].time, index: i });
          }
        }
        return troughs;
      }

      function calculateTrendSlope(values) {
        const n = values.length;
        const sumX = (n * (n - 1)) / 2;
        const sumY = values.reduce((a, b) => a + b, 0);
        const sumXY = values.reduce((sum, y, x) => sum + x * y, 0);
        const sumX2 = values.reduce((sum, _, x) => sum + x * x, 0);

        return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
      }

      function calculateSMA(values, period) {
        const result = [];
        for (let i = period - 1; i < values.length; i++) {
          const sum = values.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
          result.push(sum / period);
        }
        return result;
      }

      function calculateRSI(values, period) {
        const gains = [];
        const losses = [];

        for (let i = 1; i < values.length; i++) {
          const change = values[i] - values[i - 1];
          gains.push(change > 0 ? change : 0);
          losses.push(change < 0 ? Math.abs(change) : 0);
        }

        const result = [];
        for (let i = period - 1; i < gains.length; i++) {
          const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
          const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0) / period;
          const rs = avgGain / avgLoss;
          result.push(100 - (100 / (1 + rs)));
        }
        return result;
      }
    `;
  }

  createAlertEngineWorker() {
    return `
      let alertRules = [];
      let priceHistory = [];

      self.onmessage = function(e) {
        const { type, data, config } = e.data;

        switch(type) {
          case 'SET_ALERT_RULES':
            alertRules = data;
            self.postMessage({ type: 'ALERT_RULES_SET', data: { count: alertRules.length } });
            break;

          case 'CHECK_ALERTS':
            priceHistory.push(data);
            if (priceHistory.length > 1000) priceHistory = priceHistory.slice(-1000);

            const triggeredAlerts = checkAlerts(data, priceHistory);
            if (triggeredAlerts.length > 0) {
              self.postMessage({ type: 'ALERTS_TRIGGERED', data: triggeredAlerts });
            }
            break;

          case 'ADD_ALERT_RULE':
            alertRules.push(data);
            self.postMessage({ type: 'ALERT_RULE_ADDED', data: data });
            break;

          case 'REMOVE_ALERT_RULE':
            alertRules = alertRules.filter(rule => rule.id !== data.id);
            self.postMessage({ type: 'ALERT_RULE_REMOVED', data: data });
            break;
        }
      };

      function checkAlerts(currentData, history) {
        const triggered = [];

        alertRules.forEach(rule => {
          if (evaluateAlertRule(rule, currentData, history)) {
            triggered.push({
              ...rule,
              triggeredAt: Date.now(),
              currentPrice: currentData.close,
              message: generateAlertMessage(rule, currentData)
            });
          }
        });

        return triggered;
      }

      function evaluateAlertRule(rule, current, history) {
        switch(rule.type) {
          case 'PRICE_ABOVE':
            return current.close > rule.value;

          case 'PRICE_BELOW':
            return current.close < rule.value;

          case 'PRICE_CHANGE':
            if (history.length < 2) return false;
            const prev = history[history.length - 2];
            const changePercent = ((current.close - prev.close) / prev.close) * 100;
            return Math.abs(changePercent) > rule.value;

          case 'VOLUME_SPIKE':
            if (history.length < 20) return false;
            const avgVolume = history.slice(-20).reduce((sum, h) => sum + h.volume, 0) / 20;
            return current.volume > avgVolume * (rule.value / 100);

          case 'RSI_OVERBOUGHT':
            const rsi = calculateSimpleRSI(history.slice(-14).map(h => h.close));
            return rsi > rule.value;

          case 'RSI_OVERSOLD':
            const rsiLow = calculateSimpleRSI(history.slice(-14).map(h => h.close));
            return rsiLow < rule.value;

          case 'SUPPORT_BREAK':
            return current.close < rule.value && history[history.length - 2]?.close >= rule.value;

          case 'RESISTANCE_BREAK':
            return current.close > rule.value && history[history.length - 2]?.close <= rule.value;

          default:
            return false;
        }
      }

      function generateAlertMessage(rule, current) {
        switch(rule.type) {
          case 'PRICE_ABOVE':
            return \`Price broke above \${rule.value}! Current: \${current.close.toFixed(2)}\`;
          case 'PRICE_BELOW':
            return \`Price dropped below \${rule.value}! Current: \${current.close.toFixed(2)}\`;
          case 'VOLUME_SPIKE':
            return \`Volume spike detected! Current volume: \${current.volume.toFixed(0)}\`;
          default:
            return \`Alert triggered for \${rule.symbol}\`;
        }
      }

      function calculateSimpleRSI(prices) {
        if (prices.length < 14) return 50;

        let gains = 0, losses = 0;
        for (let i = 1; i < prices.length; i++) {
          const change = prices[i] - prices[i - 1];
          if (change > 0) gains += change;
          else losses += Math.abs(change);
        }

        const avgGain = gains / 13;
        const avgLoss = losses / 13;
        const rs = avgGain / avgLoss;
        return 100 - (100 / (1 + rs));
      }
    `;
  }

  createPerformanceTrackerWorker() {
    return `
      let performanceData = {
        startTime: Date.now(),
        requests: 0,
        errors: 0,
        latencies: [],
        dataPoints: 0,
        memoryUsage: [],
        processingTimes: []
      };

      self.onmessage = function(e) {
        const { type, data } = e.data;

        switch(type) {
          case 'TRACK_REQUEST':
            trackRequest(data);
            break;

          case 'TRACK_ERROR':
            trackError(data);
            break;

          case 'TRACK_LATENCY':
            trackLatency(data);
            break;

          case 'TRACK_PROCESSING_TIME':
            trackProcessingTime(data);
            break;

          case 'GET_PERFORMANCE_REPORT':
            const report = generatePerformanceReport();
            self.postMessage({ type: 'PERFORMANCE_REPORT', data: report });
            break;

          case 'RESET_METRICS':
            resetMetrics();
            break;
        }
      };

      function trackRequest(data) {
        performanceData.requests++;
        performanceData.dataPoints += data.dataPoints || 1;
      }

      function trackError(data) {
        performanceData.errors++;
      }

      function trackLatency(data) {
        performanceData.latencies.push(data.latency);
        if (performanceData.latencies.length > 1000) {
          performanceData.latencies = performanceData.latencies.slice(-1000);
        }
      }

      function trackProcessingTime(data) {
        performanceData.processingTimes.push(data.time);
        if (performanceData.processingTimes.length > 1000) {
          performanceData.processingTimes = performanceData.processingTimes.slice(-1000);
        }
      }

      function generatePerformanceReport() {
        const uptime = Date.now() - performanceData.startTime;
        const avgLatency = performanceData.latencies.length > 0
          ? performanceData.latencies.reduce((a, b) => a + b, 0) / performanceData.latencies.length
          : 0;
        const avgProcessingTime = performanceData.processingTimes.length > 0
          ? performanceData.processingTimes.reduce((a, b) => a + b, 0) / performanceData.processingTimes.length
          : 0;
        const errorRate = performanceData.requests > 0
          ? (performanceData.errors / performanceData.requests) * 100
          : 0;

        return {
          uptime,
          requests: performanceData.requests,
          errors: performanceData.errors,
          errorRate,
          avgLatency,
          avgProcessingTime,
          dataPoints: performanceData.dataPoints,
          requestsPerSecond: performanceData.requests / (uptime / 1000),
          dataPointsPerSecond: performanceData.dataPoints / (uptime / 1000)
        };
      }

      function resetMetrics() {
        performanceData = {
          startTime: Date.now(),
          requests: 0,
          errors: 0,
          latencies: [],
          dataPoints: 0,
          memoryUsage: [],
          processingTimes: []
        };
      }
    `;
  }

  createPatternRecognitionWorker() {
    return `
      self.onmessage = function(e) {
        const { type, data, config } = e.data;

        switch(type) {
          case 'RECOGNIZE_PATTERNS':
            const patterns = recognizeAdvancedPatterns(data, config);
            self.postMessage({ type: 'PATTERNS_RECOGNIZED', data: patterns });
            break;

          case 'ANALYZE_MARKET_STRUCTURE':
            const structure = analyzeMarketStructure(data, config);
            self.postMessage({ type: 'MARKET_STRUCTURE_ANALYZED', data: structure });
            break;

          case 'DETECT_ANOMALIES':
            const anomalies = detectAnomalies(data, config);
            self.postMessage({ type: 'ANOMALIES_DETECTED', data: anomalies });
            break;
        }
      };

      function recognizeAdvancedPatterns(data, config) {
        const patterns = [];

        // Elliott Wave patterns
        patterns.push(...detectElliottWaves(data));

        // Harmonic patterns
        patterns.push(...detectHarmonicPatterns(data));

        // Flag and Pennant patterns
        patterns.push(...detectFlagPatterns(data));

        // Cup and Handle patterns
        patterns.push(...detectCupAndHandlePatterns(data));

        // Wedge patterns
        patterns.push(...detectWedgePatterns(data));

        return patterns;
      }

      function detectElliottWaves(data) {
        const patterns = [];
        const swings = findSwingPoints(data);

        for (let i = 4; i < swings.length; i++) {
          const wave = swings.slice(i - 4, i + 1);
          if (isElliottWavePattern(wave)) {
            patterns.push({
              type: 'ELLIOTT_WAVE',
              timestamp: wave[4].timestamp,
              confidence: 0.7,
              waves: wave,
              signal: determineElliottWaveSignal(wave)
            });
          }
        }

        return patterns;
      }

      function detectHarmonicPatterns(data) {
        const patterns = [];
        const swings = findSwingPoints(data);

        for (let i = 4; i < swings.length; i++) {
          const points = swings.slice(i - 4, i + 1);

          // Check for Gartley pattern
          if (isGartleyPattern(points)) {
            patterns.push({
              type: 'GARTLEY',
              timestamp: points[4].timestamp,
              confidence: 0.8,
              signal: 'REVERSAL'
            });
          }

          // Check for Butterfly pattern
          if (isButterflyPattern(points)) {
            patterns.push({
              type: 'BUTTERFLY',
              timestamp: points[4].timestamp,
              confidence: 0.8,
              signal: 'REVERSAL'
            });
          }
        }

        return patterns;
      }

      function analyzeMarketStructure(data, config) {
        const structure = {
          trend: determineTrend(data),
          swingHighs: findSwingHighs(data),
          swingLows: findSwingLows(data),
          orderBlocks: findOrderBlocks(data),
          fairValueGaps: findFairValueGaps(data),
          liquidityPools: findLiquidityPools(data)
        };

        return structure;
      }

      function detectAnomalies(data, config) {
        const anomalies = [];

        // Price anomalies
        anomalies.push(...detectPriceAnomalies(data));

        // Volume anomalies
        anomalies.push(...detectVolumeAnomalies(data));

        // Time-based anomalies
        anomalies.push(...detectTimeAnomalies(data));

        return anomalies;
      }

      function findSwingPoints(data) {
        const swings = [];
        const lookback = 5;

        for (let i = lookback; i < data.length - lookback; i++) {
          const isSwingHigh = data.slice(i - lookback, i).every(d => d.high <= data[i].high) &&
                              data.slice(i + 1, i + lookback + 1).every(d => d.high <= data[i].high);

          const isSwingLow = data.slice(i - lookback, i).every(d => d.low >= data[i].low) &&
                             data.slice(i + 1, i + lookback + 1).every(d => d.low >= data[i].low);

          if (isSwingHigh) {
            swings.push({
              type: 'HIGH',
              price: data[i].high,
              timestamp: data[i].time,
              index: i
            });
          }

          if (isSwingLow) {
            swings.push({
              type: 'LOW',
              price: data[i].low,
              timestamp: data[i].time,
              index: i
            });
          }
        }

        return swings;
      }

      function isElliottWavePattern(waves) {
        if (waves.length !== 5) return false;

        // Basic Elliott Wave rules
        const [w1, w2, w3, w4, w5] = waves;

        // Wave 2 should not retrace more than 100% of Wave 1
        const retrace2 = Math.abs(w2.price - w1.price) / Math.abs(w1.price - w2.price);
        if (retrace2 > 1) return false;

        // Wave 3 should be longer than Wave 1
        const wave1Length = Math.abs(w2.price - w1.price);
        const wave3Length = Math.abs(w4.price - w3.price);
        if (wave3Length <= wave1Length) return false;

        // Wave 4 should not overlap Wave 1
        if ((w4.price > w1.price && w4.price < w2.price) ||
            (w4.price < w1.price && w4.price > w2.price)) return false;

        return true;
      }

      function isGartleyPattern(points) {
        if (points.length !== 5) return false;

        const [X, A, B, C, D] = points.map(p => p.price);

        // Gartley ratios
        const AB_XA = Math.abs(B - A) / Math.abs(A - X);
        const BC_AB = Math.abs(C - B) / Math.abs(B - A);
        const CD_BC = Math.abs(D - C) / Math.abs(C - B);
        const AD_XA = Math.abs(D - A) / Math.abs(A - X);

        return (AB_XA >= 0.58 && AB_XA <= 0.68) &&
               (BC_AB >= 0.38 && BC_AB <= 0.88) &&
               (CD_BC >= 1.13 && CD_BC <= 1.68) &&
               (AD_XA >= 0.75 && AD_XA <= 0.88);
      }

      function isButterflyPattern(points) {
        if (points.length !== 5) return false;

        const [X, A, B, C, D] = points.map(p => p.price);

        // Butterfly ratios
        const AB_XA = Math.abs(B - A) / Math.abs(A - X);
        const BC_AB = Math.abs(C - B) / Math.abs(B - A);
        const CD_BC = Math.abs(D - C) / Math.abs(C - B);
        const AD_XA = Math.abs(D - A) / Math.abs(A - X);

        return (AB_XA >= 0.75 && AB_XA <= 0.85) &&
               (BC_AB >= 0.38 && BC_AB <= 0.88) &&
               (CD_BC >= 1.62 && CD_BC <= 2.24) &&
               (AD_XA >= 1.25 && AD_XA <= 1.35);
      }

      // Additional helper functions would continue here...
    `;
  }

  function fetchMarketNews() {
    fetch('/api/news?limit=15')
      .then(response => response.json())
      .then(data => {
        if (data.status === 'success') {
          appState.newsItems = data.news;
          renderNewsItems();
          updateMarketSentiment(data.sentiment_summary);
        } else {
          console.error('Failed to fetch news:', data.error);
          // Fallback to mock data if API fails
          loadSampleNews();
        }
      })
      .catch(error => {
        console.error('Error fetching news:', error);
        // Fallback to mock data on network error
        loadSampleNews();
      });
  }

  function updateMarketSentiment(sentimentSummary) {
    // Update sentiment indicator in the UI
    const sentimentElement = document.getElementById('market-sentiment');
    if (sentimentElement) {
      const sentiment = sentimentSummary.sentiment_label || 'neutral';
      const score = sentimentSummary.overall_sentiment || 0;

      sentimentElement.className = `sentiment-indicator sentiment-${sentiment}`;
      sentimentElement.textContent = `Market: ${sentiment.toUpperCase()} (${score.toFixed(2)})`;
    }
  }

  function fetchAndRenderAlerts() {
    fetch(`/api/alerts?symbol=${appState.currentSymbol}`)
      .then(res => res.json())
      .then(data => {
        if (data.status === 'success' && data.alerts.length > 0) {
          appState.alerts = data.alerts.map(alert => ({
            symbol: alert.symbol,
            type: alert.type,
            message: alert.message,
            level: alert.level,
            timestamp: alert.timestamp,
            currentPrice: alert.current_price
          }));
          renderAlerts();
        } else {
          appState.alerts = [];
          renderAlerts(); // Clear UI if no alerts
        }
      })
      .catch(err => {
        console.error("Error fetching alerts:", err);
      });
  }

  function updateAlertsStatistics(stats) {
    // Update alerts statistics in the UI
    const statsElement = document.getElementById('alerts-stats');
    if (statsElement) {
      statsElement.innerHTML = `
        <div class="stat-item">
          <span>Total Alerts:</span>
          <span>${stats.total_alerts || 0}</span>
        </div>
        <div class="stat-item">
          <span>Last Hour:</span>
          <span>${stats.alerts_last_hour || 0}</span>
        </div>
        <div class="stat-item">
          <span>Monitoring:</span>
          <span class="${stats.active_monitoring ? 'status-active' : 'status-inactive'}">
            ${stats.active_monitoring ? 'Active' : 'Inactive'}
          </span>
        </div>
      `;
    }
  }

  function renderNewsItems() {
    const container = document.getElementById('newsContainer');
    container.innerHTML = '';

    if (!appState.newsItems || appState.newsItems.length === 0) {
      container.innerHTML = '<div class="no-data">No news available</div>';
      return;
    }

    appState.newsItems.forEach(item => {
      const sentimentClass = item.sentiment_class || 'news-neutral';
      const sentimentIcon = item.sentiment_class || 'sentiment-neutral';

      const newsElement = document.createElement('div');
      newsElement.className = `news-item ${sentimentClass}`;

      // Truncate description if too long
      const description = item.description && item.description.length > 100
        ? item.description.substring(0, 100) + '...'
        : item.description || '';

      newsElement.innerHTML = `
        <div class="news-title">
          <span class="news-sentiment ${sentimentIcon}"></span>
          <a href="${item.url || '#'}" target="_blank" rel="noopener">
            ${item.title}
          </a>
        </div>
        ${description ? `<div class="news-description">${description}</div>` : ''}
        <div class="news-source">
          <span>${item.source}</span>
          <span class="news-time">${new Date(item.timestamp).toLocaleTimeString()}</span>
          <span class="news-relevance">Relevance: ${((item.relevance || 0) * 100).toFixed(0)}%</span>
        </div>
      `;

      container.appendChild(newsElement);
    });
  }

  function renderAlerts() {
    const container = document.getElementById('alertsContainer');
    container.innerHTML = '';

    if (!appState.alerts || appState.alerts.length === 0) {
      container.innerHTML = '<div class="no-data">No alerts available</div>';
      return;
    }

    appState.alerts.forEach(alert => {
      const alertClass = alert.level_class || 'alert-neutral';

      const alertElement = document.createElement('div');
      alertElement.className = `alert ${alertClass}`;

      // Format additional data if available
      let additionalInfo = '';
      if (alert.additional_data) {
        const data = alert.additional_data;
        if (data.change_percent) {
          additionalInfo += `<div class="alert-detail">Change: ${data.change_percent.toFixed(2)}%</div>`;
        }
        if (data.volume_ratio) {
          additionalInfo += `<div class="alert-detail">Volume: ${data.volume_ratio.toFixed(1)}x</div>`;
        }
        if (data.rsi) {
          additionalInfo += `<div class="alert-detail">RSI: ${data.rsi.toFixed(1)}</div>`;
        }
      }

      alertElement.innerHTML = `
        <div class="alert-header">
          <strong>${alert.symbol} - ${alert.type.replace('_', ' ')}</strong>
          <span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="alert-message">${alert.message}</div>
        <div class="alert-price">Price: $${alert.current_price?.toFixed(4) || 'N/A'}</div>
        ${additionalInfo}
        <div class="alert-severity severity-${alert.severity}">${alert.severity.toUpperCase()}</div>
      `;

      container.appendChild(alertElement);
    });
  }

  // Create custom alert function
  function createCustomAlert() {
    const symbol = appState.currentSymbol;
    const alertType = prompt('Enter alert type (price_surge, volume_spike, etc.):');
    const threshold = prompt('Enter threshold value:');
    const condition = prompt('Enter condition (above/below):') || 'above';

    if (!alertType || !threshold) {
      showAlert('Please provide valid alert parameters', 'danger');
      return;
    }

    fetch('/api/alerts/create', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: symbol,
        alert_type: alertType,
        threshold: parseFloat(threshold),
        condition: condition
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.status === 'success') {
        showAlert(`Custom alert created for ${symbol}`, 'success');
      } else {
        showAlert(`Failed to create alert: ${data.error}`, 'danger');
      }
    })
    .catch(error => {
      console.error('Error creating alert:', error);
      showAlert('Error creating custom alert', 'danger');
    });
  }

  // Control alerts monitoring
  function toggleAlertsMonitoring() {
    const action = appState.alertsMonitoring ? 'stop' : 'start';

    fetch('/api/alerts/control', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ action: action })
    })
    .then(response => response.json())
    .then(data => {
      if (data.status === 'success') {
        appState.alertsMonitoring = !appState.alertsMonitoring;
        const button = document.getElementById('toggleAlertsBtn');
        if (button) {
          button.textContent = `Alerts: ${appState.alertsMonitoring ? 'ON' : 'OFF'}`;
          button.className = appState.alertsMonitoring ? 'btn success' : 'btn';
        }
        showAlert(data.message, 'success');
      } else {
        showAlert(`Failed to ${action} monitoring: ${data.error}`, 'danger');
      }
    })
    .catch(error => {
      console.error('Error controlling alerts:', error);
      showAlert('Error controlling alerts monitoring', 'danger');
    });
  }

  // Start news analysis
  function startNewsAnalysis() {
    setInterval(() => {
      fetchMarketNews();
    }, 300000); // Every 5 minutes
    fetchMarketNews(); // Initial fetch
  }


  function startAlertsMonitoring() {
    // Fetch alerts immediately
    fetchMarketAlerts();

    // Set up periodic updates (every 30 seconds)
    setInterval(() => {
      fetchMarketAlerts();
    }, 30000);
  }

  // Add to your init() function
  function init() {
    setupEventListeners();
    fetchAndPlot();
    startPerformanceTracking();
    startNewsAnalysis();      // Use real news data
    startAlertsMonitoring();  // Use real alerts data
    setupMultiChartView();
    updateSystemStatus("System: Operational");
    setInterval(fetchAndRenderAlerts, 60000);
  }

  // Add CSS styles for new elements (add to your <style> section)
  const additionalCSS = `
  .no-data {
    text-align: center;
    color: #666;
    font-style: italic;
    padding: 20px;
  }

  .news-description {
    font-size: 12px;
    color: #ccc;
    margin: 5px 0;
    line-height: 1.4;
  }

  .news-relevance {
    font-size: 10px;
    color: #999;
    margin-left: 10px;
  }

  .alert-detail {
    font-size: 11px;
    color: #aaa;
    margin: 2px 0;
  }

  .alert-severity {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 3px;
    margin-top: 5px;
    text-align: center;
  }

  .severity-high {
    background: rgba(255, 68, 68, 0.2);
    color: #ff4444;
  }

  .severity-medium {
    background: rgba(255, 170, 0, 0.2);
    color: #ffaa00;
  }

  .severity-low {
    background: rgba(136, 136, 136, 0.2);
    color: #888;
  }

  .sentiment-indicator {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
  }

  .sentiment-bullish {
    background: rgba(0, 255, 136, 0.2);
    color: #00ff88;
  }

  .sentiment-bearish {
    background: rgba(255, 68, 68, 0.2);
    color: #ff4444;
  }

  .sentiment-neutral {
    background: rgba(255, 170, 0, 0.2);
    color: #ffaa00;
  }

  #alerts-stats {
    margin-bottom: 15px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 6px;
  }

  .stat-item {
    display: flex;
    justify-content: space-between;
    margin: 5px 0;
    font-size: 12px;
  }

  .status-active {
    color: #00ff88;
  }

  .status-inactive {
    color: #ff4444;
  }
  `;

  // Add the CSS to your document
  const styleSheet = document.createElement('style');
  styleSheet.textContent = additionalCSS;
  document.head.appendChild(styleSheet);

  // ====================
  // WEBSOCKET SETUP
  // ====================

  async setupWebSocket() {
    const wsUrl = 'wss://stream.binance.com:9443/ws/btcusdt@kline_5m';

    try {
      this.websocket = new WebSocket(wsUrl);

      this.websocket.onopen = () => {
        console.log('🔗 WebSocket connected');
        this.state.wsConnected = true;
        this.state.retryCount = 0;
        this.subscribeToSymbol(this.config.currentSymbol, this.config.currentInterval);
      };

      this.websocket.onmessage = (event) => {
        this.handleWebSocketMessage(event.data);
      };

      this.websocket.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        this.state.wsConnected = false;
        this.reconnectWebSocket();
      };

      this.websocket.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        this.handleWebSocketError(error);
      };

    } catch (error) {
      console.error('❌ WebSocket setup failed:', error);
      this.fallbackToRestAPI();
    }
  }

  handleWebSocketMessage(data) {
    try {
      const message = JSON.parse(data);

      if (message.k) { // Kline data
        const kline = message.k;
        const candleData = {
          time: kline.t,
          Open: parseFloat(kline.o),
          high: parseFloat(kline.h),
          low: parseFloat(kline.l),
          close: parseFloat(kline.c),
          volume: parseFloat(kline.v)
        };

        this.processCandleData(candleData);
        this.trackPerformance('websocket_message', { dataPoints: 1 });
      }

    } catch (error) {
      console.error('❌ WebSocket message parsing error:', error);
      this.trackPerformance('websocket_error', { error: error.message });
    }
  }

  subscribeToSymbol(symbol, interval) {
    if (this.websocket && this.state.wsConnected) {
      const subscription = {
        method: "SUBSCRIBE",
        params: [`${symbol.toLowerCase()}@kline_${interval}`],
        id: Date.now()
      };

      this.websocket.send(JSON.stringify(subscription));
      this.state.activeSubscriptions.add(`${symbol}@${interval}`);

      console.log(`📡 Subscribed to ${symbol} ${interval}`);
    }
  }

  reconnectWebSocket() {
    if (this.state.retryCount < this.config.maxRetries) {
      this.state.retryCount++;
      const delay = Math.min(1000 * Math.pow(2, this.state.retryCount), 30000);

      console.log(`🔄 Reconnecting WebSocket in ${delay}ms (attempt ${this.state.retryCount})`);

      setTimeout(() => {
        this.setupWebSocket();
      }, delay);
    } else {
      console.log('❌ Max WebSocket reconnection attempts reached');
      this.fallbackToRestAPI();
    }
  }

  // ====================
  // MULTI-SYMBOL MONITORING
  // ====================

  async initializeMultiSymbolMonitoring() {
    this.state.watchlist = [
      'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT',
      'BNBUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'EOSUSDT'
    ];

    // Create multi-chart layout
    this.createMultiChartLayout();

    // Start monitoring all symbols
    for (const symbol of this.state.watchlist) {
      this.startSymbolMonitoring(symbol);
    }

    console.log(`👀 Multi-symbol monitoring started for ${this.state.watchlist.length} symbols`);
  }

  createMultiChartLayout() {
    const container = this.elements.multiChartContainer || document.body;
    container.innerHTML = `
      <div class="multi-chart-grid">
        ${this.state.watchlist.map(symbol => `
          <div class="mini-chart-container" id="chart-${symbol}">
            <div class="mini-chart-header">
              <span class="symbol-name">${symbol}</span>
              <span class="price-indicator" id="price-${symbol}">--</span>
              <span class="change-indicator" id="change-${symbol}">--</span>
            </div>
            <div class="mini-chart" id="mini-chart-${symbol}"></div>
          </div>
        `).join('')}
      </div>
    `;

    this.addMultiChartStyles();
  }

  addMultiChartStyles() {
    const style = document.createElement('style');
    style.textContent = `
      .multi-chart-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
        padding: 20px;
      }

      .mini-chart-container {
        background: rgba(20, 20, 20, 0.9);
        border-radius: 8px;
        border: 1px solid #333;
        overflow: hidden;
        height: 250px;
      }

      .mini-chart-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 15px;
        background: rgba(30, 30, 30, 0.8);
        border-bottom: 1px solid #444;
      }

      .symbol-name {
        font-weight: bold;
        color: #00d4ff;
      }

      .price-indicator {
        font-size: 14px;
        font-weight: bold;
        color: #fff;
      }

      .change-indicator {
        font-size: 12px;
        padding: 2px 6px;
        border-radius: 4px;
      }

      .change-positive {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
      }

      .change-negative {
        background: rgba(255, 68, 68, 0.2);
        color: #ff4444;
      }

      .mini-chart {
        height: calc(100% - 50px);
      }
    `;

    document.head.appendChild(style);
  }

  async startSymbolMonitoring(symbol) {
    try {
      // Fetch initial data
      const data = await this.fetchCandleData(symbol, this.config.currentInterval);

      // Create mini chart
      this.createMiniChart(symbol, data);

      // Subscribe to real-time updates
      this.subscribeToSymbol(symbol, this.config.currentInterval);

      // Set up alerts for this symbol
      this.setupSymbolAlerts(symbol);

    } catch (error) {
      console.error(`❌ Failed to start monitoring ${symbol}:`, error);
    }
  }

  createMiniChart(symbol, data) {
    const times = data.map(d => new Date(d.time));

    const trace = {
      x: times,
      open: data.map(d => d.open),
      high: data.map(d => d.high),
      low: data.map(d => d.low),
      close: data.map(d => d.close),
      type: 'candlestick',
      increasing: { line: { color: '#00ff88' } },
      decreasing: { line: { color: '#ff4444' } },
      showlegend: false
    };

    const layout = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#e0e0e0', size: 10 },
      margin: { l: 30, r: 30, t: 10, b: 30 },
      xaxis: {
        showgrid: false,
        showticklabels: false,
        rangeslider: { visible: false }
      },
      yaxis: {
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        side: 'right',
        tickfont: { size: 9 }
      }
    };

    const config = {
      displayModeBar: false,
      responsive: true
    };

    Plotly.newPlot(`mini-chart-${symbol}`, [trace], layout, config);

    // Update price indicators
    this.updatePriceIndicators(symbol, data);
  }

  updatePriceIndicators(symbol, data) {
    const latestPrice = data[data.length - 1];
    const previousPrice = data[data.length - 2];

    const priceElement = document.getElementById(`price-${symbol}`);
    const changeElement = document.getElementById(`change-${symbol}`);

    if (priceElement) {
      priceElement.textContent = latestPrice.close.toFixed(2);
    }

    if (changeElement && previousPrice) {
      const change = latestPrice.close - previousPrice.close;
      const changePercent = (change / previousPrice.close) * 100;

      changeElement.textContent = `${change >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
      changeElement.className = `change-indicator ${change >= 0 ? 'change-positive' : 'change-negative'}`;
    }
  }

  // ====================
  // AUTOMATION ENGINE
  // ====================

  async startAutomationEngine() {
    // Portfolio tracker
    this.startPortfolioTracking();

    // Market scanner
    this.startMarketScanner();

    // News sentiment analysis
    this.startNewsSentimentAnalysis();

    // Advanced pattern recognition
    this.startAdvancedPatternRecognition();

    // Risk management system
    this.startRiskManagement();

    // Backtesting engine
    this.startBacktestingEngine();

    console.log('🤖 Ultra-Automation Engine Started');
  }

  startPortfolioTracking() {
    setInterval(async () => {
      try {
        const portfolio = await this.calculatePortfolioMetrics();
        this.updatePortfolioDisplay(portfolio);

        // Check for rebalancing opportunities
        const rebalanceSignals = this.checkRebalancing(portfolio);
        if (rebalanceSignals.length > 0) {
          this.handleRebalancingSignals(rebalanceSignals);
        }

      } catch (error) {
        console.error('❌ Portfolio tracking error:', error);
      }
    }, 60000); // Every minute
  }

  async calculatePortfolioMetrics() {
    // Simulate portfolio data - in real implementation, fetch from API
    return {
      totalValue: 10000,
      dailyPnL: 150.75,
      totalPnL: 1250.30,
      winRate: 68.5,
      sharpeRatio: 1.45,
      maxDrawdown: -5.2,
      positions: [
        { symbol: 'BTCUSDT', value: 5000, weight: 50, pnl: 120 },
        { symbol: 'ETHUSDT', value: 3000, weight: 30, pnl: 45 },
        { symbol: 'ADAUSDT', value: 2000, weight: 20, pnl: -15 }
      ]
    };
  }

  startMarketScanner() {
    setInterval(async () => {
      try {
        const marketConditions = await this.scanMarketConditions();
        this.analyzeMarketOpportunities(marketConditions);

      } catch (error) {
        console.error('❌ Market scanner error:', error);
      }
    }, 30000); // Every 30 seconds
  }

  async scanMarketConditions() {
    const conditions = {
      overallSentiment: 'BULLISH',
      volatility: 'MEDIUM',
      volume: 'HIGH',
      topGainers: ['ADAUSDT', 'DOTUSDT', 'LINKUSDT'],
      topLosers: ['XRPUSDT', 'LTCUSDT'],
      breakouts: ['BTCUSDT', 'ETHUSDT'],
      newsEvents: this.getLatestNews()
    };

    return conditions;
  }

  // ====================
  // ENHANCED DATA PROCESSING
  // ====================

  async processCandleData(candleData) {
    const startTime = performance.now();

    try {
      // Send to data processor worker
      this.workers.get('dataProcessor')?.postMessage({
        type: 'PROCESS_CANDLES',
        data: [candleData],
        config: { period: 20 }
      });

      // Send to technical analysis worker
      this.workers.get('technicalAnalysis')?.postMessage({
        type: 'ANALYZE_PATTERNS',
        data: [candleData],
        config: { sensitivity: 0.8 }
      });

      // Send to alert engine
      this.workers.get('alertEngine')?.postMessage({
        type: 'CHECK_ALERTS',
        data: candleData
      });

      // Cache the data
      this.cacheData(candleData);

      // Update main chart
      this.updateMainChart(candleData);

      // Track performance
      const processingTime = performance.now() - startTime;
      this.trackPerformance('data_processing', { time: processingTime });

    } catch (error) {
      console.error('❌ Data processing error:', error);
      this.trackPerformance('processing_error', { error: error.message });
    }
  }

  cacheData(data) {
    const key = `${this.config.currentSymbol}_${this.config.currentInterval}`;

    if (!this.state.dataCache.has(key)) {
      this.state.dataCache.set(key, {
        data: [],
        lastUpdate: Date.now()
      });
    }

    const cached = this.state.dataCache.get(key);
    cached.data.push(data);
    cached.lastUpdate = Date.now();

    // Keep only last 1000 candles
    if (cached.data.length > 1000) {
      cached.data = cached.data.slice(-1000);
    }

    // Clean expired cache
    this.cleanExpiredCache();
  }

  cleanExpiredCache() {
    const now = Date.now();
    for (const [key, value] of this.state.dataCache.entries()) {
      if (now - value.lastUpdate > this.config.cacheExpiry) {
        this.state.dataCache.delete(key);
      }
    }
  }

  // ====================
  // WORKER MESSAGE HANDLING
  // ====================

  handleWorkerMessage(workerName, data) {
    switch (data.type) {
      case 'CANDLES_PROCESSED':
        this.handleProcessedCandles(data.data);
        break;

      case 'INDICATORS_CALCULATED':
        this.handleCalculatedIndicators(data.data);
        break;

      case 'PATTERNS_ANALYZED':
        this.handleAnalyzedPatterns(data.data);
        break;

      case 'SIGNALS_GENERATED':
        this.handleGeneratedSignals(data.data);
        break;

      case 'ALERTS_TRIGGERED':
        this.handleTriggeredAlerts(data.data);
        break;

      case 'PERFORMANCE_REPORT':
        this.handlePerformanceReport(data.data);
        break;

      case 'PATTERNS_RECOGNIZED':
        this.handleRecognizedPatterns(data.data);
        break;

      case 'MARKET_STRUCTURE_ANALYZED':
        this.handleMarketStructure(data.data);
        break;

      case 'ANOMALIES_DETECTED':
        this.handleDetectedAnomalies(data.data);
        break;

      default:
        console.log(`📨 Unhandled message from ${workerName}:`, data);
    }
  }

  handleTriggeredAlerts(alerts) {
    alerts.forEach(alert => {
      this.showAlert(alert);
      this.state.alertHistory.push(alert);

      // Play sound notification
      this.playAlertSound(alert.type);

      // Send browser notification
      this.sendBrowserNotification(alert);

      // Log to console
      console.log(`🚨 ALERT: ${alert.message}`);
    });

    this.updateAlertsPanel();
  }

  showAlert(alert) {
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${alert.signal.toLowerCase()}`;
    alertElement.innerHTML = `
      <div class="alert-header">
        <strong>${alert.symbol || this.config.currentSymbol}</strong>
        <span class="alert-time">${new Date(alert.triggeredAt).toLocaleTimeString()}</span>
      </div>
      <div class="alert-message">${alert.message}</div>
      <div class="alert-price">Price: ${alert.currentPrice?.toFixed(2) || '--'}</div>
    `;

    // Add to alerts container
    const alertsContainer = this.elements.alertsPanel || document.body;
    alertsContainer.appendChild(alertElement);

    // Auto-remove after 10 seconds
    setTimeout(() => {
      alertElement.remove();
    }, 10000);
  }

  playAlertSound(alertType) {
    try {
      const audio = new Audio();
      audio.volume = 0.5;

      switch (alertType) {
        case 'PRICE_ABOVE':
        case 'RESISTANCE_BREAK':
          audio.src = 'data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMGJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMGJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMG';
          break;
        case 'PRICE_BELOW':
        case 'SUPPORT_BREAK':
          audio.src = 'data:audio/wav;base64,UklGRnIBAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YU4BAADBhYmFcF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMGJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMGJHfH8N2QQAoUXrTp66hVFApGn+DyvmUgBTiH0fPTgjMG';
          break;