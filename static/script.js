// Enhanced Binance Candlestick Chart Script
let currentSymbol = "BTCUSDT";
let currentInterval = "5m";
let refreshInterval;
let isLoading = false;
let lastUpdateTime = null;
let retryCount = 0;
const MAX_RETRIES = 3;

// Performance optimization: Cache DOM elements
const elements = {
  symbolSelect: document.getElementById("symbolSelect"),
  intervalSelect: document.getElementById("intervalSelect"),
  chart: document.getElementById("chart"),
  lastUpdate: document.getElementById("lastUpdate"),
  errorMessage: document.getElementById("errorMessage")
};

// Event listeners with enhanced functionality
elements.symbolSelect?.addEventListener("change", (e) => {
  currentSymbol = e.target.value;
  resetRefreshInterval();
  fetchAndPlot();
});

elements.intervalSelect?.addEventListener("change", (e) => {
  currentInterval = e.target.value;
  resetRefreshInterval();
  fetchAndPlot();
});

// Dynamic refresh rates based on interval
function getRefreshRate(interval) {
  const refreshRates = {
    '1m': 5000,    // 5 seconds for 1-minute charts
    '3m': 10000,   // 10 seconds for 3-minute charts
    '5m': 15000,   // 15 seconds for 5-minute charts
    '15m': 30000,  // 30 seconds for 15-minute charts
    '30m': 60000,  // 1 minute for 30-minute charts
    '1h': 120000,  // 2 minutes for 1-hour charts
    '2h': 240000,  // 4 minutes for 2-hour charts
    '4h': 300000,  // 5 minutes for 4-hour charts
    '6h': 360000,  // 6 minutes for 6-hour charts
    '8h': 480000,  // 8 minutes for 8-hour charts
    '12h': 720000, // 12 minutes for 12-hour charts
    '1d': 1800000, // 30 minutes for daily charts
    '3d': 3600000, // 1 hour for 3-day charts
    '1w': 7200000, // 2 hours for weekly charts
    '1M': 14400000 // 4 hours for monthly charts
  };
  return refreshRates[interval] || 15000; // Default 15 seconds
}

// Reset refresh interval with new rate
function resetRefreshInterval() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
  const refreshRate = getRefreshRate(currentInterval);
  refreshInterval = setInterval(fetchAndPlot, refreshRate);
  console.log(`Refresh rate set to ${refreshRate / 1000} seconds for ${currentInterval} interval`);
}

// Enhanced error handling and retry logic
function handleError(error, context = '') {
  console.error(`Error ${context}:`, error);
  retryCount++;
  
  if (retryCount < MAX_RETRIES) {
    console.log(`Retrying... Attempt ${retryCount + 1}/${MAX_RETRIES}`);
    setTimeout(() => fetchAndPlot(), 2000 * retryCount); // Exponential backoff
  } else {
    showErrorMessage(`Failed to load data after ${MAX_RETRIES} attempts. Please check your connection.`);
    retryCount = 0; // Reset for next manual attempt
  }
}

// Show error message to user
function showErrorMessage(message) {
  if (elements.chart) {
    elements.chart.innerHTML = `
      <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        height: 100%; 
        color: #ff6666; 
        background: rgba(255, 0, 0, 0.1); 
        border: 1px solid #ff4444; 
        border-radius: 8px; 
        margin: 20px; 
        padding: 20px;
        text-align: center;
      ">
        <div>
          <h3>Connection Error</h3>
          <p>${message}</p>
          <button onclick="fetchAndPlot()" style="
            background: #ff4444; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            margin-top: 10px;
          ">Try Again</button>
        </div>
      </div>
    `;
  }
}

// Show loading state
function showLoading() {
  if (elements.chart && !elements.chart.querySelector('.plotly-graph-div')) {
    elements.chart.innerHTML = `
      <div style="
        display: flex; 
        align-items: center; 
        justify-content: center; 
        height: 100%; 
        color: #00d4ff;
      ">
        <div style="text-align: center;">
          <div style="
            width: 40px; 
            height: 40px; 
            border: 4px solid #333; 
            border-top: 4px solid #00d4ff; 
            border-radius: 50%; 
            animation: spin 1s linear infinite; 
            margin: 0 auto 15px;
          "></div>
          <p>Loading chart data...</p>
        </div>
      </div>
      <style>
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      </style>
    `;
  }
}

// Enhanced fetchAndPlot function
function fetchAndPlot() {
  // Prevent multiple simultaneous requests
  if (isLoading) {
    console.log("Request already in progress, skipping...");
    return;
  }

  isLoading = true;
  showLoading();

  // Add timestamp to prevent caching issues
  const timestamp = Date.now();
  const url = `/candles?symbol=${currentSymbol}&interval=${currentInterval}&_t=${timestamp}`;

  fetch(url)
    .then(res => {
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      return res.json();
    })
    .then(data => {
      if (data.error) {
        throw new Error(data.error);
      }

      // Validate data
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error("No data received from server");
      }

      // Reset retry count on successful fetch
      retryCount = 0;
      
      plotCandlestickChart(data);
      updateLastUpdateTime();
      
      isLoading = false;
    })
    .catch(error => {
      isLoading = false;
      handleError(error, 'fetching data');
    });
}

// Separated plotting logic for better maintainability
function plotCandlestickChart(data) {
  try {
    const times = data.map(d => new Date(d.time));
    
    // Enhanced candlestick trace
    const trace = {
      x: times,
      open: data.map(d => d.open),
      high: data.map(d => d.high),
      low: data.map(d => d.low),
      close: data.map(d => d.close),
      type: 'candlestick',
      name: currentSymbol,
      increasing: { 
        line: { color: '#00ff88', width: 1 },
        fillcolor: 'rgba(0, 255, 136, 0.3)'
      },
      decreasing: { 
        line: { color: '#ff4444', width: 1 },
        fillcolor: 'rgba(255, 68, 68, 0.3)'
      },
      line: { width: 1 },
      whiskerwidth: 0.8,
      xaxis: 'x',
      yaxis: 'y'
    };

    // Enhanced layout with better styling
    const layout = {
      dragmode: 'pan',
      showlegend: false,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(10,10,10,0.8)',
      font: { color: '#e0e0e0', family: 'Segoe UI' },
      
      xaxis: {
        rangeslider: { 
          visible: true,
          bgcolor: 'rgba(40,40,40,0.8)',
          bordercolor: '#555',
          borderwidth: 1,
          thickness: 0.08
        },
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        gridwidth: 1,
        tickfont: { color: '#ccc', size: 11 },
        title: { text: 'Time', font: { color: '#ccc', size: 12 } },
        showspikes: true,
        spikecolor: '#00d4ff',
        spikethickness: 1,
        spikedash: 'dot'
      },
      
      yaxis: {
        showgrid: true,
        gridcolor: 'rgba(255,255,255,0.1)',
        gridwidth: 1,
        tickfont: { color: '#ccc', size: 11 },
        title: { text: 'Price (USDT)', font: { color: '#ccc', size: 12 } },
        side: 'right',
        showspikes: true,
        spikecolor: '#00d4ff',
        spikethickness: 1,
        spikedash: 'dot'
      },
      
      margin: { l: 60, r: 80, t: 40, b: 60 },
      
      hovermode: 'x unified',
      hoverlabel: {
        bgcolor: 'rgba(0,0,0,0.8)',
        bordercolor: '#00d4ff',
        font: { color: '#fff', size: 12 }
      },
      
      // Add title with current symbol and interval
      title: {
        text: `${currentSymbol} - ${currentInterval.toUpperCase()} Chart`,
        font: { color: '#e0e0e0', size: 16 },
        x: 0.5,
        xanchor: 'center'
      }
    };

    // Enhanced config with better controls
    const config = {
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      modeBarButtonsToAdd: [
        {
          name: 'Reset Zoom',
          icon: {
            width: 500,
            height: 500,
            path: 'M250,100 L350,200 L250,300 L150,200 Z'
          },
          click: function(gd) {
            Plotly.relayout(gd, {
              'xaxis.autorange': true,
              'yaxis.autorange': true
            });
          }
        }
      ],
      scrollZoom: true,
      responsive: true,
      toImageButtonOptions: {
        format: 'png',
        filename: `${currentSymbol}_${currentInterval}_${new Date().toISOString().split('T')[0]}`,
        height: 600,
        width: 1000,
        scale: 1
      }
    };

    // Use newPlot for first render, relayout for updates to improve performance
    if (!elements.chart.querySelector('.plotly-graph-div')) {
      Plotly.newPlot('chart', [trace], layout, config);
    } else {
      Plotly.react('chart', [trace], layout, config);
    }

  } catch (error) {
    console.error('Error plotting chart:', error);
    showErrorMessage('Failed to render chart. Please try refreshing the page.');
  }
}

// Update last update time display
function updateLastUpdateTime() {
  if (elements.lastUpdate) {
    const now = new Date();
    elements.lastUpdate.textContent = `Last updated: ${now.toLocaleTimeString()}`;
    lastUpdateTime = now;
  }
}

// Add price change indicator
function addPriceChangeIndicator(data) {
  if (data && data.length >= 2) {
    const current = data[data.length - 1].close;
    const previous = data[data.length - 2].close;
    const change = current - previous;
    const changePercent = ((change / previous) * 100).toFixed(2);
    
    const indicator = document.getElementById('priceChange');
    if (indicator) {
      indicator.innerHTML = `
        <span style="color: ${change >= 0 ? '#00ff88' : '#ff4444'}">
          ${change >= 0 ? '↗' : '↘'} ${change.toFixed(2)} (${changePercent}%)
        </span>
      `;
    }
  }
}

// Visibility API for pausing updates when tab is not active
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    console.log('Tab hidden, pausing updates');
    if (refreshInterval) {
      clearInterval(refreshInterval);
    }
  } else {
    console.log('Tab visible, resuming updates');
    resetRefreshInterval();
    fetchAndPlot(); // Immediate update when tab becomes visible
  }
});

// Handle window resize for responsive chart
window.addEventListener('resize', debounce(() => {
  if (elements.chart.querySelector('.plotly-graph-div')) {
    Plotly.Plots.resize('chart');
  }
}, 250));

// Debounce utility function
function debounce(func, wait) {
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

// Cleanup function for when page unloads
window.addEventListener('beforeunload', () => {
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
  // Ctrl/Cmd + R for manual refresh
  if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
    e.preventDefault();
    fetchAndPlot();
  }
  
  // Space for pause/resume
  if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
    e.preventDefault();
    toggleAutoRefresh();
  }
});

// Toggle auto refresh functionality
let autoRefreshPaused = false;
function toggleAutoRefresh() {
  autoRefreshPaused = !autoRefreshPaused;
  
  if (autoRefreshPaused) {
    clearInterval(refreshInterval);
    console.log('Auto-refresh paused');
  } else {
    resetRefreshInterval();
    console.log('Auto-refresh resumed');
  }
}

// Connection status monitoring
let isOnline = navigator.onLine;
window.addEventListener('online', () => {
  isOnline = true;
  console.log('Connection restored');
  if (!autoRefreshPaused) {
    resetRefreshInterval();
    fetchAndPlot();
  }
});

window.addEventListener('offline', () => {
  isOnline = false;
  console.log('Connection lost');
  if (refreshInterval) {
    clearInterval(refreshInterval);
  }
  showErrorMessage('No internet connection. Please check your network.');
});

// Initialize the application
function init() {
  console.log(`Initializing chart for ${currentSymbol} with ${currentInterval} interval`);
  fetchAndPlot();
  resetRefreshInterval();
}

// Start the application
init();
