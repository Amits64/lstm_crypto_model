# news.py - Real Market News Provider
import requests
import json
import time
from typing import List, Dict
import logging
from textblob import TextBlob
import feedparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoNewsProvider:
    """
    Fetches real cryptocurrency news from multiple sources and analyzes sentiment
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Real news sources (using free APIs and RSS feeds)
        self.sources = {
            'coindesk': {
                'rss': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'type': 'rss'
            },
            'cointelegraph': {
                'rss': 'https://cointelegraph.com/rss',
                'type': 'rss'
            },
            'decrypt': {
                'rss': 'https://decrypt.co/feed',
                'type': 'rss'
            },
            'bitcoinmagazine': {
                'rss': 'https://bitcoinmagazine.com/.rss/full/',
                'type': 'rss'
            },
            'coinjournal': {
                'rss': 'https://coinjournal.net/feed/',
                'type': 'rss'
            },
            'bitcoinist': {
                'rss': 'https://bitcoinist.com/feed/',
                'type': 'rss'
            }
        }

        # Crypto-related keywords for relevance scoring
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'cryptocurrency',
            'blockchain', 'defi', 'nft', 'binance', 'coinbase', 'trading',
            'altcoin', 'hodl', 'mining', 'staking', 'dao', 'web3', 'metaverse',
            'regulation', 'sec', 'fed', 'cbdc', 'stablecoin', 'usdt', 'usdc',
            'dogecoin', 'cardano', 'solana', 'polkadot', 'chainlink', 'litecoin',
            'ripple', 'xrp', 'bnb', 'ada', 'sol', 'dot', 'link', 'ltc',
            'smart contract', 'dapp', 'yield farming', 'liquidity', 'amm'
        ]

        # Cache for avoiding duplicate requests
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

    def calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score using TextBlob
        Returns: float between -1 (very negative) and 1 (very positive)
        """
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0

    def calculate_relevance(self, title: str, description: str = "") -> float:
        """
        Calculate how relevant the news is to cryptocurrency
        Returns: float between 0 and 1
        """
        text = f"{title} {description}".lower()
        keyword_count = sum(1 for keyword in self.crypto_keywords if keyword in text)

        # Weight certain keywords higher
        high_value_keywords = ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi']
        high_value_count = sum(2 for keyword in high_value_keywords if keyword in text)

        total_score = keyword_count + high_value_count
        max_possible = len(self.crypto_keywords) + len(high_value_keywords) * 2

        return min(total_score / max_possible * 3, 1.0)  # Scale and cap at 1.0

    def fetch_rss_news(self, source_name: str, rss_url: str, limit: int = 15) -> List[Dict]:
        """Fetch news from RSS feed"""
        try:
            # Check cache
            cache_key = f"rss_{source_name}"
            if cache_key in self.cache:
                cache_time, cached_data = self.cache[cache_key]
                if time.time() - cache_time < self.cache_duration:
                    return cached_data

            logger.info(f"Fetching RSS news from {source_name}")

            # Set timeout for RSS fetch
            feed = feedparser.parse(rss_url)

            if not feed.entries:
                logger.warning(f"No entries found for {source_name}")
                return []

            news_items = []
            for entry in feed.entries[:limit]:
                try:
                    # Extract and clean content
                    title = entry.get('title', 'No title').strip()
                    if not title or title == 'No title':
                        continue

                    description = entry.get('summary', entry.get('description', ''))

                    # Clean HTML tags and normalize text
                    description = re.sub(r'<[^>]+>', '', description)
                    description = re.sub(r'\s+', ' ', description).strip()
                    description = description[:300] + '...' if len(description) > 300 else description

                    # Parse publish date
                    pub_date = entry.get('published_parsed')
                    if pub_date:
                        timestamp = time.mktime(pub_date) * 1000
                    else:
                        timestamp = int(time.time() * 1000)

                    # Skip very old news (older than 7 days)
                    if timestamp < (time.time() - 7 * 24 * 3600) * 1000:
                        continue

                    # Calculate sentiment and relevance
                    content_for_analysis = f"{title} {description}"
                    sentiment = self.calculate_sentiment(content_for_analysis)
                    relevance = self.calculate_relevance(title, description)

                    # Only include crypto-relevant news
                    if relevance > 0.2:  # Stricter relevance threshold
                        news_items.append({
                            'title': title,
                            'description': description,
                            'source': source_name.title(),
                            'url': entry.get('link', ''),
                            'timestamp': int(timestamp),
                            'sentiment': sentiment,
                            'relevance': relevance
                        })

                except Exception as e:
                    logger.warning(f"Error processing RSS entry from {source_name}: {e}")
                    continue

            # Cache the results
            self.cache[cache_key] = (time.time(), news_items)
            logger.info(f"Fetched {len(news_items)} relevant news items from {source_name}")
            return news_items

        except Exception as e:
            logger.error(f"Error fetching RSS from {source_name}: {e}")
            return []

    def get_latest_news(self, limit: int = 20, min_relevance: float = 0.3) -> List[Dict]:
        """
        Fetch latest cryptocurrency news from all sources
        """
        all_news = []

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = []

            for source_name, config in self.sources.items():
                if config['type'] == 'rss':
                    future = executor.submit(
                        self.fetch_rss_news,
                        source_name,
                        config['rss'],
                        limit // len(self.sources) + 3
                    )
                    futures.append(future)

            # Collect results with timeout
            for future in as_completed(futures, timeout=30):
                try:
                    news_items = future.result(timeout=10)
                    all_news.extend(news_items)
                except Exception as e:
                    logger.error(f"Error getting news from source: {e}")

        if not all_news:
            logger.warning("No news items retrieved from any source")
            return []

        # Filter by relevance and remove duplicates
        seen_titles = set()
        filtered_news = []

        for item in all_news:
            # Normalize title for duplicate detection
            normalized_title = re.sub(r'[^\w\s]', '', item['title'].lower()).strip()

            if (item['relevance'] >= min_relevance and
                    normalized_title not in seen_titles and
                    len(normalized_title) > 10):  # Skip very short titles
                seen_titles.add(normalized_title)
                filtered_news.append(item)

        # Sort by timestamp (newest first) and relevance
        filtered_news.sort(key=lambda x: (x['timestamp'], x['relevance']), reverse=True)

        return filtered_news[:limit]

    def get_sentiment_summary(self, news_items: List[Dict]) -> Dict:
        """
        Calculate overall market sentiment from news
        """
        if not news_items:
            return {
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'total_items': 0
            }

        # Weight sentiment by relevance
        weighted_sentiments = []
        for item in news_items:
            weight = item.get('relevance', 1.0)
            weighted_sentiments.append(item['sentiment'] * weight)

        overall_sentiment = sum(weighted_sentiments) / len(weighted_sentiments)

        positive_count = sum(1 for s in weighted_sentiments if s > 0.1)
        negative_count = sum(1 for s in weighted_sentiments if s < -0.1)
        neutral_count = len(weighted_sentiments) - positive_count - negative_count

        if overall_sentiment > 0.2:
            sentiment_label = 'bullish'
        elif overall_sentiment < -0.2:
            sentiment_label = 'bearish'
        else:
            sentiment_label = 'neutral'

        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_label': sentiment_label,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'total_items': len(news_items)
        }


# Global news provider instance
news_provider = CryptoNewsProvider()


def get_crypto_news(limit: int = 15) -> Dict:
    """
    Main function to get cryptocurrency news
    Returns formatted news data for the frontend
    """
    try:
        # Get latest news from real sources
        news_items = news_provider.get_latest_news(limit=limit)

        if not news_items:
            return {
                'status': 'error',
                'error': 'No news items available',
                'news': [],
                'sentiment_summary': {},
                'last_updated': int(time.time() * 1000)
            }

        # Format for frontend
        formatted_news = []
        for item in news_items:
            sentiment_class = 'bullish' if item['sentiment'] > 0.2 else (
                'bearish' if item['sentiment'] < -0.2 else 'neutral'
            )

            formatted_news.append({
                'title': item['title'],
                'description': item['description'],
                'source': item['source'],
                'url': item['url'],
                'timestamp': item['timestamp'],
                'sentiment': item['sentiment'],
                'sentiment_class': sentiment_class,
                'relevance': item['relevance']
            })

        # Get sentiment summary
        sentiment_summary = news_provider.get_sentiment_summary(news_items)

        return {
            'status': 'success',
            'news': formatted_news,
            'sentiment_summary': sentiment_summary,
            'last_updated': int(time.time() * 1000)
        }

    except Exception as e:
        logger.error(f"Error getting crypto news: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'news': [],
            'sentiment_summary': {},
            'last_updated': int(time.time() * 1000)
        }


if __name__ == "__main__":
    # Test the news provider
    result = get_crypto_news(limit=10)
    print(json.dumps(result, indent=2))