#!/usr/bin/env python3
import asyncio
import random
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import yfinance as yf

# Install the polygon-api-client if not already installed
try:
    from polygon import RESTClient
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'polygon-api-client'])
    from polygon import RESTClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UniversalMind")


@dataclass
class Concept:
    id: str
    examples: List[Dict]
    confidence: float = 0.0
    domain: str = "unknown"


@dataclass
class Rule:
    antecedent: str
    consequent: str
    confidence: float = 0.6


class UniversalCognitiveCore:
    def __init__(self, mind_id: str = "wanderer-001"):
        self.mind_id = mind_id
        self.iteration = 0
        self.running = False

        self.concepts: Dict[str, Concept] = {}
        self.rules: List[Rule] = []
        self.short_term_memory: List[Dict] = []
        self.cross_domain_mappings: Dict = {}

        self.metrics = {
            "concepts_formed": 0,
            "rules_learned": 0,
            "transfers_made": 0,
            "goals_generated": 0,
            "total_observations": 0
        }

        logger.info(f"🌌 Universal Mind {mind_id} awakened — now wandering the markets")

    def ingest(self, observation: Dict[str, Any], domain: str = "raw") -> Dict:
        self.iteration += 1
        self.metrics["total_observations"] += 1

        self.short_term_memory.append(observation)
        if len(self.short_term_memory) > 300:
            self.short_term_memory.pop(0)

        concept_id = self._form_concept(observation, domain)
        new_rules = self._infer_rules(observation)
        for rule in new_rules:
            self.rules.append(rule)
            self.metrics["rules_learned"] += 1

        if len({c.domain for c in self.concepts.values()}) > 1:
            self._attempt_cross_domain_transfer(domain)

        if self.iteration % 12 == 0:
            self._generate_autonomous_goals(observation)

        return {
            "iteration": self.iteration,
            "concept_formed": concept_id,
            "new_rules": len(new_rules),
            "current_concepts": len(self.concepts),
            "symbol": observation.get("symbol", "unknown")
        }

    def _form_concept(self, obs: Dict, domain: str) -> str:
        """Proper signature-based concept formation (no more snowflake syndrome)"""
        signature = frozenset(
            (k, round(v, 3) if isinstance(v, float) else v)
            for k, v in obs.items() if not k.startswith("_") and v is not None
        )
        concept_id = f"concept_{hash(signature) % 999999}"

        if concept_id not in self.concepts:
            self.concepts[concept_id] = Concept(
                id=concept_id,
                examples=[obs],
                confidence=0.3,
                domain=domain
            )
            self.metrics["concepts_formed"] += 1
            logger.info(f"🧩 New concept born: {concept_id} in {domain} | {obs.get('symbol')}")
            return concept_id

        self.concepts[concept_id].examples.append(obs)
        self.concepts[concept_id].confidence = min(1.0, self.concepts[concept_id].confidence + 0.18)
        return concept_id

    def _infer_rules(self, obs: Dict) -> List[Rule]:
        rules = []
        keys = [k for k in obs.keys() if isinstance(obs[k], (int, float))]

        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                v1, v2 = obs[k1], obs[k2]
                if v1 > 0.75 * v2:
                    rules.append(Rule(f"{k1}_strong", f"{k2}_elevated", 0.75))
        return rules[:4]

    def _attempt_cross_domain_transfer(self, current_domain: str):
        for other in {c.domain for c in self.concepts.values()}:
            if other != current_domain:
                key = f"{current_domain}→{other}"
                if key not in self.cross_domain_mappings:
                    self.cross_domain_mappings[key] = True
                    self.metrics["transfers_made"] += 1
                    logger.info(f"🔄 Cross-domain transfer: {current_domain} → {other}")

    def _generate_autonomous_goals(self, obs: Dict):
        goal = {
            "id": f"goal_{self.metrics['goals_generated']+1}",
            "description": f"Decode why {list(obs.keys())[:4]} move together in {obs.get('symbol', 'market')}",
            "priority": random.uniform(0.65, 0.97)
        }
        self.metrics["goals_generated"] += 1
        logger.info(f"🌱 New goal spawned: {goal['description']}")

    def introspect(self) -> Dict:
        return {
            "mind_id": self.mind_id,
            "age": self.iteration,
            "concepts": len(self.concepts),
            "rules": len(self.rules),
            "domains": len({c.domain for c in self.concepts.values()}),
            "transfers": self.metrics["transfers_made"],
            "goals": self.metrics["goals_generated"],
            "memory": len(self.short_term_memory),
            "status": "wandering" if self.iteration > 30 else "awakening"
        }

    def get_concepts(self) -> Dict[str, Concept]:
        """Returns all learned concepts."""
        return self.concepts

    def get_rules(self) -> List[Rule]:
        """Returns all learned rules."""
        return self.rules

    def get_short_term_memory(self) -> List[Dict]:
        """Returns the contents of the short-term memory."""
        return self.short_term_memory


# ============= MARKET WANDERER =============

MARKET_SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD",
    "SMCI", "AVGO", "ORCL", "CRM", "ADBE", "NFLX", "INTC", "ARM",
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"
]


def fetch_yfinance(symbol: str) -> Dict:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d", interval="1m")
        if not hist.empty:
            latest = hist.iloc[-1]
            return {
                "datetime": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"]),
                "volume": int(latest["Volume"]),
                "symbol": symbol
            }
        return {"error": "no_data"}
    except Exception as e:
        logger.debug(f"yfinance failed {symbol}: {e}")
        return {"error": "yf_error"}


def fetch_polygon(symbol: str, client: RESTClient) -> Dict:
    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=3)

        aggs = client.get_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="minute",
            from_=from_date.strftime("%Y-%m-%d"),
            to=to_date.strftime("%Y-%m-%d"),
            limit=10
        )

        if aggs:
            latest = aggs[-1]
            return {
                "datetime": datetime.fromtimestamp(latest.timestamp / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                "open": latest.open,
                "high": latest.high,
                "low": latest.low,
                "close": latest.close,
                "volume": latest.volume,
                "symbol": symbol
            }
        return {"error": "no_data"}
    except Exception as e:
        logger.debug(f"Polygon failed {symbol}: {e}")
        return {"error": "poly_error"}


async def fetch_market_data(symbol: str, polygon_client: Optional[RESTClient] = None) -> Dict:
    # yfinance first (fast & free)
    data = await asyncio.to_thread(fetch_yfinance, symbol)
    if not data.get("error"):
        return data

    # Polygon fallback
    if polygon_client:
        data = await asyncio.to_thread(fetch_polygon, symbol, polygon_client)
        if not data.get("error"):
            return data

    return {"error": "fetch_failed"}


def transform_market_data(raw: Dict) -> Dict:
    transformed = {"domain": "finance"}

    for k in ["open", "high", "low", "close", "volume"]:
        if k in raw:
            transformed[k] = raw[k]

    if "close" in transformed and "open" in transformed:
        transformed["price_change"] = transformed["close"] - transformed["open"]

    if "datetime" in raw:
        transformed["timestamp"] = raw["datetime"]
    if "symbol" in raw:
        transformed["symbol"] = raw["symbol"]

    return {k: v for k, v in transformed.items() if v is not None}


async def wander_the_market(
    symbols: List[str],
    polygon_key: str,
    mind: UniversalCognitiveCore,
    delay_seconds: int = 18,
    max_iterations: int = 2000
):
    logger.info(f"🌍 Mind is now wandering {len(symbols)} assets...")

    polygon_client = RESTClient(polygon_key) if polygon_key and polygon_key != "YOUR_KEY" else None

    for i in range(max_iterations):
        symbol = random.choice(symbols)
        logger.info(f"[{i+1:04d}] Wandering → {symbol}")

        raw = await fetch_market_data(symbol, polygon_client)

        if not raw.get("error"):
            obs = transform_market_data(raw)
            result = mind.ingest(obs, domain="finance")

            if (i + 1) % 5 == 0:
                logger.info(f"🧠 Introspection: {mind.introspect()}")
        else:
            logger.warning(f"Failed to fetch {symbol}")

        await asyncio.sleep(delay_seconds)


async def main():
    POLYGON_API_KEY = "uoeAfk5WIefD0SYQ2rhIfRqGdRRQKUcI"   # free at polygon.io

    mind = UniversalCognitiveCore("market_wanderer")

    await wander_the_market(
        MARKET_SYMBOLS,
        POLYGON_API_KEY,
        mind,
        delay_seconds=18,
        max_iterations=50  # Changed to 50 for a shorter demo
    )

    logger.info("\n" + "="*70)
    logger.info("FINAL MIND STATE AFTER WANDERING")
    logger.info("="*70)
    print(json.dumps(mind.introspect(), indent=2))


# --- FIX FOR COLAB/JUPYTER ENVIRONMENTS ---
# In Colab/Jupyter, an event loop is usually already running.
# Calling asyncio.run() directly from a cell will raise a RuntimeError.
# Instead, await the main() function directly if not in a __main__ context.
if __name__ == "__main__":
    asyncio.run(main())
