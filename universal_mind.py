#!/usr/bin/env python3
"""
Universal Cognitive Core v0.2 - Multi-Symbol Market Data Integration
Improved version that cycles through multiple symbols for better learning
"""

import asyncio
import time
import logging
import json
import random
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

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
    """
    The complete agnostic cognitive system.
    No domains hardcoded. No sensors. No actuators.
    Just pure intelligence waiting for data.
    """
    def __init__(self, mind_id: str = "mind-001"):
        self.mind_id = mind_id
        self.iteration = 0
        self.running = False
        
        # Memory systems (all in RAM, grow naturally)
        self.concepts: Dict[str, Concept] = {}
        self.rules: List[Rule] = []
        self.short_term_memory: List[Dict] = []
        self.cross_domain_mappings: Dict = {}
        
        # Metrics that prove it's alive and growing
        self.metrics = {
            "concepts_formed": 0,
            "rules_learned": 0,
            "transfers_made": 0,
            "goals_generated": 0,
            "total_observations": 0
        }
        
        logger.info(f"🌌 Universal Mind {mind_id} awakened — completely domain-agnostic")
    
    def ingest(self, observation: Dict[str, Any], domain: str = "raw") -> Dict:
        """
        The only public API you will ever need.
        Feed it ANY JSON-like dict. It will do the rest.
        """
        self.iteration += 1
        self.metrics["total_observations"] += 1
        
        self.short_term_memory.append(observation)
        if len(self.short_term_memory) > 200:
            self.short_term_memory.pop(0)
        
        # 1. Learn raw patterns
        concept_id = self._form_concept(observation, domain)
        
        # 2. Extract and learn rules
        new_rules = self._infer_rules(observation)
        for rule in new_rules:
            self.rules.append(rule)
            self.metrics["rules_learned"] += 1
        
        # 3. Cross-domain transfer (if we have multiple domains)
        if len({c.domain for c in self.concepts.values()}) > 1:
            self._attempt_cross_domain_transfer(domain)
        
        # 4. Generate goals when interesting
        if self.iteration % 15 == 0:
            self._generate_autonomous_goals(observation)
        
        return {
            "iteration": self.iteration,
            "concept_formed": concept_id,
            "new_rules": len(new_rules),
            "current_concepts": len(self.concepts),
            "urgency": "high" if any(v > 0.8 for v in observation.values() if isinstance(v, (int, float))) else "normal"
        }
    
    def _form_concept(self, obs: Dict, domain: str) -> Optional[str]:
        """Simple but effective concept formation"""
        # Create signature based on value ranges
        signature_parts = []
        for k, v in obs.items():
            if not k.startswith("_") and k not in ["timestamp", "symbol"]:
                if isinstance(v, float):
                    # Bin numeric values for pattern matching
                    binned = round(v / 10) * 10  # Round to nearest 10
                    signature_parts.append((k, binned))
        
        signature = frozenset(signature_parts)
        
        # Check if this pattern exists
        for concept_id, concept in self.concepts.items():
            if concept.domain == domain:
                # Simple similarity check
                concept_sig = frozenset(
                    (k, round(v / 10) * 10 if isinstance(v, float) else v)
                    for ex in concept.examples[:1]  # Check first example
                    for k, v in ex.items()
                    if not k.startswith("_") and k not in ["timestamp", "symbol"]
                )
                if len(signature & concept_sig) / max(len(signature), len(concept_sig), 1) > 0.7:
                    # Similar pattern found - strengthen it
                    concept.examples.append(obs)
                    concept.confidence = min(1.0, concept.confidence + 0.1)
                    return concept_id
        
        # New pattern - create new concept
        concept_id = f"concept_{len(self.concepts)+1}"
        self.concepts[concept_id] = Concept(
            id=concept_id,
            examples=[obs],
            confidence=0.3,
            domain=domain
        )
        self.metrics["concepts_formed"] += 1
        logger.info(f"🧩 New concept born: {concept_id} in {domain}")
        return concept_id
    
    def _infer_rules(self, obs: Dict) -> List[Rule]:
        """Very simple rule induction from data"""
        rules = []
        keys = [k for k in obs.keys() if k not in ['timestamp', 'domain', 'symbol']]
        
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1, k2 = keys[i], keys[j]
                v1, v2 = obs[k1], obs[k2]
                
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    if v1 > 0.7 * v2:
                        rules.append(Rule(f"{k1}_high", f"{k2}_elevated", 0.7))
        
        return rules[:3]  # keep it sane
    
    def _attempt_cross_domain_transfer(self, current_domain: str):
        """When it sees the same pattern in different domains → magic happens"""
        for other_domain in {c.domain for c in self.concepts.values()}:
            if other_domain == current_domain:
                continue
            
            key = f"{current_domain}→{other_domain}"
            if key not in self.cross_domain_mappings:
                self.cross_domain_mappings[key] = True
                self.metrics["transfers_made"] += 1
                logger.info(f"🔄 Cross-domain insight: {current_domain} patterns → {other_domain}")
    
    def _generate_autonomous_goals(self, obs: Dict):
        """It starts wanting things"""
        goal = {
            "id": f"goal_{self.metrics['goals_generated']+1}",
            "description": f"Understand why {list(obs.keys())[:3]} co-vary across domains",
            "priority": random.uniform(0.6, 0.95)
        }
        self.metrics["goals_generated"] += 1
        logger.info(f"🌱 New autonomous goal: {goal['description']}")
    
    def introspect(self) -> Dict:
        """The mirror test"""
        return {
            "mind_id": self.mind_id,
            "age": self.iteration,
            "concepts": len(self.concepts),
            "rules": len(self.rules),
            "domains_seen": len({c.domain for c in self.concepts.values()}),
            "transfers": self.metrics["transfers_made"],
            "goals": self.metrics["goals_generated"],
            "memory_size": len(self.short_term_memory),
            "status": "growing" if self.iteration > 50 else "awakening"
        }


# ============= MARKET DATA INTEGRATION =============

async def fetch_market_data(symbol: str, api_key: str, interval: str) -> Dict:
    """Fetches market data from Twelve Data API"""
    base_url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": api_key,
        "outputsize": 1  # Only get the latest bar
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error fetching data for {symbol}: {e.response.status_code}")
        if e.response.status_code == 429:
            logger.warning("Rate limit hit. Implementing back-off strategy.")
            return {"error": "rate_limit", "status_code": 429}
        return {"error": "http_error", "status_code": e.response.status_code}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {symbol}: {e}")
        return {"error": "request_exception"}
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from response for {symbol}")
        return {"error": "json_decode_error"}


async def transform_market_data(raw_data: dict, symbol: str) -> dict:
    """Converts raw market data into UniversalCognitiveCore format"""
    transformed_data = {"domain": f"finance_{symbol}"}  # Separate domain per symbol
    
    # Extract and convert numerical features
    for key in ['open', 'high', 'low', 'close', 'volume']:
        if key in raw_data and raw_data[key] is not None:
            try:
                transformed_data[key] = float(raw_data[key])
            except ValueError:
                logger.warning(f"Could not convert {raw_data[key]} to float for {key}")
                transformed_data[key] = None
        else:
            transformed_data[key] = None
    
    # Calculate additional features
    if transformed_data.get('close') is not None and transformed_data.get('open') is not None:
        transformed_data['price_change'] = transformed_data['close'] - transformed_data['open']
        transformed_data['volatility'] = transformed_data.get('high', 0) - transformed_data.get('low', 0)
    
    # Include timestamp and symbol
    if 'datetime' in raw_data:
        transformed_data['timestamp'] = raw_data['datetime']
    transformed_data['symbol'] = symbol
    
    # Filter out None values
    return {k: v for k, v in transformed_data.items() if v is not None}


async def stream_multi_symbol_data(
    symbols: List[str],
    api_key: str,
    interval: str,
    delay_seconds: int,
    mind_instance,
    max_iterations: int = 50,
    max_retries: int = 3
):
    """Streams data from multiple symbols for better learning diversity"""
    logger.info(f"🚀 Starting MULTI-SYMBOL market data stream")
    logger.info(f"   Symbols: {', '.join(symbols)}")
    logger.info(f"   Interval: {interval} | Max Iterations: {max_iterations}\n")
    
    consecutive_rate_limit_hits = 0
    retry_delay = delay_seconds
    iteration = 0
    symbol_index = 0
    
    while iteration < max_iterations:
        # Cycle through symbols
        symbol = symbols[symbol_index % len(symbols)]
        symbol_index += 1
        
        logger.info(f"📊 Fetching data for {symbol} (iteration {iteration + 1}/{max_iterations})...")
        data = await fetch_market_data(symbol, api_key, interval)
        
        if data and not data.get("error"):
            # Successfully fetched data
            consecutive_rate_limit_hits = 0
            retry_delay = delay_seconds
            
            if "values" in data and len(data["values"]) > 0:
                latest_data = data["values"][0]
                logger.info(f"   [{symbol}] Time: {latest_data.get('datetime')} | "
                          f"Close: ${latest_data.get('close')} | "
                          f"Volume: {latest_data.get('volume')}")
                
                # Transform the data for ingestion
                transformed_market_data = await transform_market_data(latest_data, symbol)
                
                # Ingest into UniversalCognitiveCore
                ingestion_result = mind_instance.ingest(transformed_market_data, domain=f"finance_{symbol}")
                logger.info(f"   ✅ Ingested: {ingestion_result['concept_formed']} | "
                          f"Concepts: {ingestion_result['current_concepts']} | "
                          f"Rules: +{ingestion_result['new_rules']}\n")
                
                # Show introspection periodically
                if (iteration + 1) % 10 == 0:
                    introspection = mind_instance.introspect()
                    logger.info(f"🧠 === MIND INTROSPECTION ===")
                    logger.info(f"   Age: {introspection['age']} observations")
                    logger.info(f"   Concepts: {introspection['concepts']}")
                    logger.info(f"   Rules: {introspection['rules']}")
                    logger.info(f"   Domains: {introspection['domains_seen']}")
                    logger.info(f"   Transfers: {introspection['transfers']}")
                    logger.info(f"   Goals: {introspection['goals']}")
                    logger.info(f"   Status: {introspection['status'].upper()}\n")
            else:
                logger.warning(f"No 'values' found in response for {symbol}: {data}")
        
        elif data and data.get("error") == "rate_limit":
            consecutive_rate_limit_hits += 1
            if consecutive_rate_limit_hits > max_retries:
                logger.error(f"Exceeded max retries for rate limit. Stopping stream.")
                break
            
            retry_delay *= 2  # Exponential back-off
            logger.warning(f"⚠️ Rate limit hit. Retrying in {retry_delay} seconds...\n")
        
        elif data and data.get("error"):
            logger.error(f"Error fetching data for {symbol}: {data.get('error')}\n")
            # Don't break - try next symbol
        
        else:
            logger.error(f"Unexpected empty response for {symbol}\n")
        
        iteration += 1
        await asyncio.sleep(retry_delay)
    
    logger.info(f"🏁 Market data stream completed after {iteration} iterations")


async def main():
    """Main execution function"""
    # API Configuration
    API_KEY = "20986c52844e4e1ba6156404bcb52bb0"
    
    # Use multiple symbols to ensure diversity even when markets are closed
    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
    INTERVAL = "1min"
    DELAY_SECONDS = 10  # Delay between requests
    MAX_ITERATIONS = 50  # Reduced from 200 to be more reasonable
    
    logger.info("=" * 70)
    logger.info("🎯 UNIVERSAL COGNITIVE CORE - MULTI-SYMBOL MARKET DATA")
    logger.info("=" * 70)
    logger.info(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"⚠️  Note: Markets may be closed. System will learn from available data.")
    logger.info("=" * 70 + "\n")
    
    # Initialize the UniversalCognitiveCore
    mind = UniversalCognitiveCore("market_mind")
    logger.info(f"✅ UniversalCognitiveCore '{mind.mind_id}' initialized.\n")
    
    # Start the multi-symbol market data stream
    await stream_multi_symbol_data(
        SYMBOLS,
        API_KEY,
        INTERVAL,
        DELAY_SECONDS,
        mind,
        MAX_ITERATIONS
    )
    
    # Final introspection
    logger.info("\n" + "=" * 70)
    logger.info("📊 FINAL MIND STATE:")
    logger.info("=" * 70)
    final_state = mind.introspect()
    print(json.dumps(final_state, indent=2))
    
    # Show some concepts
    logger.info(f"\n🧩 CONCEPTS FORMED:")
    for concept_id, concept in list(mind.concepts.items())[:5]:
        logger.info(f"   {concept_id}: {concept.domain} (confidence: {concept.confidence:.2f}, examples: {len(concept.examples)})")
    
    logger.info("\n🎉 Demo complete!")


if __name__ == "__main__":
    asyncio.run(main())
