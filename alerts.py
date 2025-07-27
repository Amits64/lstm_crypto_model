# alerts.py - Real Market Alerts System
import json
import time
import threading
from typing import List, Dict, Optional, Callable
import logging
from binance.client import Client
import numpy as np
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketAlertsEngine:
    """
    Real-time market alerts engine that monitors multiple cryptocurrencies
    and generates intelligent alerts based on price action, volume, and technical indicators
    """

    def __init__(self, symbols: List[str] = None):
        self.client = Client()

        # Default symbols to monitor
        self.symbols = symbols or [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
            'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 'MATICUSDT', 'AVAXUSDT'
        ]

        # Data storage
        self.price_data = defaultdict(lambda: deque(maxlen=200))  # Store last 200 price points
        self.volume_data = defaultdict(lambda: deque(maxlen=50))  # Store last 50 volume points
        self.alerts = deque(maxlen=100)  # Store last 100 alerts

        # Alert configurations
        self.alert_configs = {
            'price_surge': {'threshold': 2.5, 'timeframe': 15},  # 2.5% in 15 minutes
            'price_drop': {'threshold': -2.0, 'timeframe': 15},  # -2% in 15 minutes
            'volume_spike': {'threshold': 2.0, 'timeframe': 5},  # 2x volume in 5 minutes
            'breakout': {'threshold': 1.5, 'period': 20},  # 1.5% above 20-period high
            'breakdown': {'threshold': -1.5, 'period': 20},  # 1.5% below 20-period low
            'volatility_spike': {'threshold': 3.0, 'period': 14},  # 3x average volatility
            'support_resistance': {'strength': 0.5, 'period': 50},  # Support/resistance levels
            'rsi_extremes': {'oversold': 25, 'overbought': 75},  # RSI extreme levels
            'unusual_activity': {'volume_mult': 3, 'price_mult': 1.5}  # Unusual market activity
        }

        # Threading
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        # Performance tracking
        self.last_update = {}
        self.alert_count = defaultdict(int)

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_volatility(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate price volatility"""
        if len(prices) < period:
            return None

        returns = np.diff(np.log(prices[-period:]))
        return np.std(returns) * np.sqrt(period)

    def detect_support_resistance(self, prices: List[float], period: int = 50) -> Dict:
        """Detect support and resistance levels"""
        if len(prices) < period:
            return {'support': None, 'resistance': None}

        recent_prices = prices[-period:]
        current_price = prices[-1]

        # Simple support/resistance detection
        support = np.percentile(recent_prices, 25)
        resistance = np.percentile(recent_prices, 75)

        return {
            'support': support,
            'resistance': resistance,
            'near_support': abs(current_price - support) / current_price < 0.02,
            'near_resistance': abs(current_price - resistance) / current_price < 0.02
        }

    def generate_price_surge_alert(self, symbol: str, current_price: float,
                                   price_change: float, timeframe: int) -> Dict:
        """Generate price surge alert"""
        return {
            'id': f"surge_{symbol}_{int(time.time())}",
            'type': 'PRICE_SURGE',
            'symbol': symbol,
            'level': 'buy',
            'title': f'{symbol} Price Surge',
            'message': f'Price increased by {price_change:.2f}% in the last {timeframe} minutes',
            'current_price': current_price,
            'change_percent': price_change,
            'timestamp': int(time.time() * 1000),
            'severity': 'high' if abs(price_change) > 5 else 'medium'
        }

    def generate_volume_spike_alert(self, symbol: str, current_price: float,
                                    volume_ratio: float) -> Dict:
        """Generate volume spike alert"""
        return {
            'id': f"volume_{symbol}_{int(time.time())}",
            'type': 'VOLUME_SPIKE',
            'symbol': symbol,
            'level': 'buy',
            'title': f'{symbol} Volume Spike',
            'message': f'Volume spike detected ({volume_ratio:.1f}x average volume)',
            'current_price': current_price,
            'volume_ratio': volume_ratio,
            'timestamp': int(time.time() * 1000),
            'severity': 'high' if volume_ratio > 5 else 'medium'
        }

    def generate_breakout_alert(self, symbol: str, current_price: float,
                                level_type: str, level_price: float) -> Dict:
        """Generate breakout/breakdown alert"""
        direction = 'above' if level_type == 'resistance' else 'below'
        alert_type = 'BREAKOUT' if level_type == 'resistance' else 'BREAKDOWN'
        level = 'buy' if level_type == 'resistance' else 'sell'

        return {
            'id': f"breakout_{symbol}_{int(time.time())}",
            'type': alert_type,
            'symbol': symbol,
            'level': level,
            'title': f'{symbol} {alert_type.title()}',
            'message': f'Price broke {direction} key {level_type} level at {level_price:.4f}',
            'current_price': current_price,
            'level_price': level_price,
            'timestamp': int(time.time() * 1000),
            'severity': 'high'
        }

    def generate_rsi_alert(self, symbol: str, current_price: float, rsi: float) -> Dict:
        """Generate RSI extreme alert"""
        condition = 'oversold' if rsi < 30 else 'overbought'
        level = 'buy' if condition == 'oversold' else 'sell'

        return {
            'id': f"rsi_{symbol}_{int(time.time())}",
            'type': 'RSI_EXTREME',
            'symbol': symbol,
            'level': level,
            'title': f'{symbol} RSI {condition.title()}',
            'message': f'RSI indicates {condition} condition (RSI: {rsi:.1f})',
            'current_price': current_price,
            'rsi': rsi,
            'timestamp': int(time.time() * 1000),
            'severity': 'medium'
        }

    def generate_volatility_alert(self, symbol: str, current_price: float,
                                  volatility: float, avg_volatility: float) -> Dict:
        """Generate volatility spike alert"""
        return {
            'id': f"volatility_{symbol}_{int(time.time())}",
            'type': 'VOLATILITY_SPIKE',
            'symbol': symbol,
            'level': 'neutral',
            'title': f'{symbol} High Volatility',
            'message': f'Volatility spike detected ({volatility / avg_volatility:.1f}x normal)',
            'current_price': current_price,
            'volatility_ratio': volatility / avg_volatility,
            'timestamp': int(time.time() * 1000),
            'severity': 'medium'
        }

    def analyze_symbol(self, symbol: str) -> List[Dict]:
        """Analyze a symbol and generate alerts"""
        alerts = []

        try:
            # Get recent price data
            if symbol not in self.price_data or len(self.price_data[symbol]) < 20:
                return alerts

            prices = list(self.price_data[symbol])
            volumes = list(self.volume_data[symbol]) if self.volume_data[symbol] else []
            current_price = prices[-1]

            # 1. Price surge/drop detection
            if len(prices) >= 15:
                price_15min_ago = prices[-15]
                price_change = ((current_price - price_15min_ago) / price_15min_ago) * 100

                if price_change >= self.alert_configs['price_surge']['threshold']:
                    alerts.append(self.generate_price_surge_alert(
                        symbol, current_price, price_change, 15
                    ))
                elif price_change <= self.alert_configs['price_drop']['threshold']:
                    alert = self.generate_price_surge_alert(
                        symbol, current_price, price_change, 15
                    )
                    alert['type'] = 'PRICE_DROP'
                    alert['level'] = 'sell'
                    alert['title'] = f'{symbol} Price Drop'
                    alert['message'] = f'Price decreased by {abs(price_change):.2f}% in the last 15 minutes'
                    alerts.append(alert)

            # 2. Volume spike detection
            if len(volumes) >= 10:
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[-10:-1])  # Exclude current volume
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio >= self.alert_configs['volume_spike']['threshold']:
                        alerts.append(self.generate_volume_spike_alert(
                            symbol, current_price, volume_ratio
                        ))

            # 3. Support/Resistance breakouts
            if len(prices) >= 50:
                sr_levels = self.detect_support_resistance(prices, 50)
                if sr_levels['resistance'] and current_price > sr_levels['resistance'] * 1.015:
                    alerts.append(self.generate_breakout_alert(
                        symbol, current_price, 'resistance', sr_levels['resistance']
                    ))
                elif sr_levels['support'] and current_price < sr_levels['support'] * 0.985:
                    alerts.append(self.generate_breakout_alert(
                        symbol, current_price, 'support', sr_levels['support']
                    ))

            # 4. RSI extremes
            if len(prices) >= 30:
                rsi = self.calculate_rsi(prices, 14)
                if rsi is not None:
                    if rsi <= self.alert_configs['rsi_extremes']['oversold']:
                        alerts.append(self.generate_rsi_alert(symbol, current_price, rsi))
                    elif rsi >= self.alert_configs['rsi_extremes']['overbought']:
                        alerts.append(self.generate_rsi_alert(symbol, current_price, rsi))

            # 5. Volatility spikes
            if len(prices) >= 50:
                current_volatility = self.calculate_volatility(prices[-14:], 14)
                avg_volatility = self.calculate_volatility(prices[-50:-14], 14)

                if (current_volatility and avg_volatility and
                        current_volatility > avg_volatility * self.alert_configs['volatility_spike']['threshold']):
                    alerts.append(self.generate_volatility_alert(
                        symbol, current_price, current_volatility, avg_volatility
                    ))

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

        return alerts

    def update_market_data(self):
        """Update market data for all symbols"""
        try:
            # Get 1-minute klines for all symbols
            for symbol in self.symbols:
                try:
                    klines = self.client.get_klines(
                        symbol=symbol,
                        interval=Client.KLINE_INTERVAL_1MINUTE,
                        limit=5
                    )

                    for kline in klines[-3:]:  # Take last 3 candles
                        timestamp = int(kline[0])

                        # Skip if we already have this timestamp
                        if (symbol in self.last_update and
                                timestamp <= self.last_update.get(symbol, 0)):
                            continue

                        price = float(kline[4])  # Close price
                        volume = float(kline[5])

                        with self.lock:
                            self.price_data[symbol].append(price)
                            self.volume_data[symbol].append(volume)
                            self.last_update[symbol] = timestamp

                except Exception as e:
                    logger.warning(f"Error updating data for {symbol}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error updating market data: {e}")

    def monitor_markets(self):
        """Main monitoring loop"""
        logger.info("Starting market monitoring...")

        while self.running:
            try:
                # Update market data
                self.update_market_data()

                # Analyze each symbol for alerts
                new_alerts = []
                for symbol in self.symbols:
                    symbol_alerts = self.analyze_symbol(symbol)
                    new_alerts.extend(symbol_alerts)

                # Add new alerts to queue
                with self.lock:
                    for alert in new_alerts:
                        self.alerts.append(alert)
                        self.alert_count[alert['type']] += 1
                        logger.info(f"Generated alert: {alert['type']} for {alert['symbol']}")

                # Sleep before next analysis
                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error

    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_markets, daemon=True)
            self.monitor_thread.start()
            logger.info("Market monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Market monitoring stopped")

    def get_recent_alerts(self, limit: int = 20, symbol: str = None) -> List[Dict]:
        """Get recent alerts"""
        with self.lock:
            alerts_list = list(self.alerts)

        # Filter by symbol if specified
        if symbol:
            alerts_list = [alert for alert in alerts_list if alert['symbol'] == symbol]

        # Sort by timestamp (newest first) and limit
        alerts_list.sort(key=lambda x: x['timestamp'], reverse=True)
        return alerts_list[:limit]

    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        with self.lock:
            total_alerts = len(self.alerts)
            recent_alerts = [a for a in self.alerts if a['timestamp'] > (time.time() - 3600) * 1000]

            return {
                'total_alerts': total_alerts,
                'alerts_last_hour': len(recent_alerts),
                'alert_types': dict(self.alert_count),
                'monitored_symbols': len(self.symbols),
                'active_monitoring': self.running
            }


# Global alerts engine instance
alerts_engine = MarketAlertsEngine()


def get_market_alerts(limit: int = 20, symbol: str = None) -> Dict:
    """
    Main function to get market alerts
    Returns formatted alerts data for the frontend
    """
    try:
        # Ensure monitoring is started
        if not alerts_engine.running:
            alerts_engine.start_monitoring()

        # Get recent alerts
        alerts_list = alerts_engine.get_recent_alerts(limit=limit, symbol=symbol)

        # Format for frontend
        formatted_alerts = []
        for alert in alerts_list:
            # Map alert levels to CSS classes
            level_class = {
                'buy': 'alert-buy',
                'sell': 'alert-sell',
                'neutral': 'alert-neutral'
            }.get(alert['level'], 'alert-neutral')

            formatted_alerts.append({
                'id': alert['id'],
                'type': alert['type'],
                'symbol': alert['symbol'],
                'title': alert['title'],
                'message': alert['message'],
                'level': alert['level'],
                'level_class': level_class,
                'current_price': alert['current_price'],
                'timestamp': alert['timestamp'],
                'severity': alert.get('severity', 'medium'),
                'additional_data': {
                    k: v for k, v in alert.items()
                    if k not in ['id', 'type', 'symbol', 'title', 'message', 'level', 'current_price', 'timestamp']
                }
            })

        # Get statistics
        stats = alerts_engine.get_alert_statistics()

        return {
            'status': 'success',
            'alerts': formatted_alerts,
            'statistics': stats,
            'last_updated': int(time.time() * 1000)
        }

    except Exception as e:
        logger.error(f"Error getting market alerts: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'alerts': [],
            'statistics': {},
            'last_updated': int(time.time() * 1000)
        }


def create_custom_alert(symbol: str, alert_type: str, threshold: float,
                        condition: str = 'above') -> Dict:
    """
    Create a custom user-defined alert
    """
    try:
        custom_alert = {
            'id': f"custom_{symbol}_{alert_type}_{int(time.time())}",
            'type': 'CUSTOM_ALERT',
            'symbol': symbol,
            'alert_type': alert_type,
            'threshold': threshold,
            'condition': condition,
            'created_at': int(time.time() * 1000),
            'status': 'active'
        }

        # You could store custom alerts in a database or file
        # For now, we'll just return the alert configuration

        return {
            'status': 'success',
            'message': f'Custom alert created for {symbol}',
            'alert': custom_alert
        }

    except Exception as e:
        logger.error(f"Error creating custom alert: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def stop_monitoring():
    """Stop the alerts monitoring"""
    alerts_engine.stop_monitoring()
    return {'status': 'success', 'message': 'Monitoring stopped'}


def start_monitoring():
    """Start the alerts monitoring"""
    alerts_engine.start_monitoring()
    return {'status': 'success', 'message': 'Monitoring started'}


if __name__ == "__main__":
    # Test the alerts engine
    print("Starting alerts engine test...")

    # Start monitoring
    alerts_engine.start_monitoring()

    # Wait a bit for data to accumulate
    time.sleep(60)

    # Get alerts
    result = get_market_alerts(limit=10)
    print(json.dumps(result, indent=2))

    # Stop monitoring
    alerts_engine.stop_monitoring()