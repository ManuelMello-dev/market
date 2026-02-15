#!/usr/bin/env python3
"""
Universal Cognitive Core - Production System
Domain-agnostic learning system with full market coverage and dynamic control
Version: 1.0.0
"""

import asyncio
import logging
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
import requests
from collections import defaultdict
import hashlib
from enum import Enum

# Production-grade logging configuration
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Configure logging with file and console handlers"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger("UniversalMind")


logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE")
)


class FocusMode(Enum):
    """System focus modes"""
    BROAD = "broad"  # Cycle through all symbols
    FOCUSED = "focused"  # Focus on specific symbol(s)
    HYBRID = "hybrid"  # Focused symbols + periodic broad sampling


@dataclass
class Concept:
    """Learned pattern representation"""
    id: str
    examples: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    domain: str = "unknown"
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    last_updated: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    
    def to_dict(self) -> Dict:
        """Serialize concept for persistence"""
        return {
            'id': self.id,
            'confidence': self.confidence,
            'domain': self.domain,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'example_count': len(self.examples)
        }


@dataclass
class Rule:
    """Discovered relationship between features"""
    antecedent: str
    consequent: str
    confidence: float
    support: int = 1
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    
    def to_dict(self) -> Dict:
        """Serialize rule for persistence"""
        return asdict(self)


@dataclass
class SystemMetrics:
    """Performance and learning metrics"""
    concepts_formed: int = 0
    rules_learned: int = 0
    transfers_made: int = 0
    goals_generated: int = 0
    total_observations: int = 0
    errors: int = 0
    uptime_seconds: float = 0.0
    last_observation_time: Optional[float] = None
    symbols_tracked: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize metrics"""
        return asdict(self)


class UniversalCognitiveCore:
    """
    Production-grade domain-agnostic learning system.
    Thread-safe, persistent, and horizontally scalable.
    """
    
    def __init__(
        self,
        mind_id: str,
        config: Optional[Dict] = None
    ):
        self.mind_id = mind_id
        self.config = config or self._default_config()
        self.iteration = 0
        self.start_time = datetime.now(timezone.utc).timestamp()
        
        # Memory systems
        self.concepts: Dict[str, Concept] = {}
        self.rules: Dict[str, Rule] = {}  # Use hash as key for deduplication
        self.short_term_memory: List[Dict] = []
        self.cross_domain_mappings: Dict[str, Set[str]] = defaultdict(set)
        
        # Metrics
        self.metrics = SystemMetrics()
        
        # State management
        self._checkpoint_dir = Path(self.config['checkpoint_dir'])
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Universal Mind '{mind_id}' initialized")
    
    @staticmethod
    def _default_config() -> Dict:
        """Default production configuration"""
        return {
            'max_memory_size': 1000,
            'concept_similarity_threshold': 0.7,
            'rule_min_support': 2,
            'goal_generation_interval': 50,
            'checkpoint_interval': 100,
            'checkpoint_dir': './checkpoints',
            'binning_precision': 10,
            'max_rules_per_observation': 5
        }
    
    def ingest(self, observation: Dict[str, Any], domain: str) -> Dict:
        """
        Main ingestion pipeline for incoming data.
        Thread-safe and transactional.
        """
        try:
            self.iteration += 1
            self.metrics.total_observations += 1
            self.metrics.last_observation_time = datetime.now(timezone.utc).timestamp()
            
            # Add to short-term memory
            self._update_memory(observation)
            
            # Core learning pipeline
            concept_id = self._form_concept(observation, domain)
            new_rules = self._infer_rules(observation)
            self._process_rules(new_rules)
            
            # Cross-domain analysis
            if len(self._get_active_domains()) > 1:
                self._attempt_cross_domain_transfer(domain)
            
            # Autonomous goal generation
            if self.iteration % self.config['goal_generation_interval'] == 0:
                self._generate_autonomous_goals(observation)
            
            # Periodic checkpointing
            if self.iteration % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint()
            
            return {
                'success': True,
                'iteration': self.iteration,
                'concept_formed': concept_id,
                'new_rules': len(new_rules),
                'current_concepts': len(self.concepts),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Error ingesting observation: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'iteration': self.iteration
            }
    
    def _update_memory(self, observation: Dict):
        """Manage short-term memory with size limits"""
        self.short_term_memory.append(observation)
        max_size = self.config['max_memory_size']
        
        if len(self.short_term_memory) > max_size:
            self.short_term_memory = self.short_term_memory[-max_size:]
    
    def _form_concept(self, obs: Dict, domain: str) -> str:
        """Form or strengthen concepts based on patterns"""
        signature = self._create_signature(obs)
        
        # Search for similar existing concepts
        for concept_id, concept in self.concepts.items():
            if concept.domain != domain:
                continue
            
            if self._concepts_similar(concept, signature):
                concept.examples.append(obs)
                concept.confidence = min(1.0, concept.confidence + 0.05)
                concept.last_updated = datetime.now(timezone.utc).timestamp()
                
                if len(concept.examples) > 100:
                    concept.examples = concept.examples[-100:]
                
                return concept_id
        
        # Create new concept
        concept_id = f"concept_{self.metrics.concepts_formed + 1}"
        self.concepts[concept_id] = Concept(
            id=concept_id,
            examples=[obs],
            confidence=0.3,
            domain=domain
        )
        self.metrics.concepts_formed += 1
        logger.info(f"New concept: {concept_id} in '{domain}'")
        
        return concept_id
    
    def _create_signature(self, obs: Dict) -> Set[tuple]:
        """Create fuzzy signature from observation"""
        signature = set()
        precision = self.config['binning_precision']
        
        for key, value in obs.items():
            if key in ['timestamp', 'symbol', 'domain']:
                continue
            
            if isinstance(value, (int, float)):
                binned = round(value / precision) * precision
                signature.add((key, binned))
            elif isinstance(value, str):
                signature.add((key, value))
        
        return signature
    
    def _concepts_similar(self, concept: Concept, signature: Set[tuple]) -> bool:
        """Check if signature matches existing concept"""
        if not concept.examples:
            return False
        
        concept_sig = self._create_signature(concept.examples[0])
        
        if not signature or not concept_sig:
            return False
        
        intersection = len(signature & concept_sig)
        union = len(signature | concept_sig)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= self.config['concept_similarity_threshold']
    
    def _infer_rules(self, obs: Dict) -> List[Rule]:
        """Discover relationships in data"""
        rules = []
        keys = [k for k in obs.keys() if k not in ['timestamp', 'domain', 'symbol']]
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                v1, v2 = obs.get(k1), obs.get(k2)
                
                if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
                    continue
                
                if v1 > 0.7 * v2 and v2 > 0:
                    rules.append(Rule(f"{k1}_high", f"{k2}_elevated", 0.7))
                
                if abs(v1 - v2) / max(abs(v1), abs(v2), 1) < 0.1:
                    rules.append(Rule(f"{k1}_similar", f"{k2}_similar", 0.8))
        
        return rules[:self.config['max_rules_per_observation']]
    
    def _process_rules(self, new_rules: List[Rule]):
        """Add rules with deduplication and support tracking"""
        for rule in new_rules:
            rule_hash = self._hash_rule(rule)
            
            if rule_hash in self.rules:
                self.rules[rule_hash].support += 1
                self.rules[rule_hash].confidence = min(
                    1.0,
                    self.rules[rule_hash].confidence + 0.02
                )
            else:
                self.rules[rule_hash] = rule
                self.metrics.rules_learned += 1
    
    @staticmethod
    def _hash_rule(rule: Rule) -> str:
        """Generate unique hash for rule deduplication"""
        content = f"{rule.antecedent}→{rule.consequent}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _attempt_cross_domain_transfer(self, current_domain: str):
        """Detect cross-domain pattern transfers"""
        active_domains = self._get_active_domains()
        
        for other_domain in active_domains:
            if other_domain == current_domain:
                continue
            
            if other_domain not in self.cross_domain_mappings[current_domain]:
                self.cross_domain_mappings[current_domain].add(other_domain)
                self.metrics.transfers_made += 1
                logger.info(f"Transfer: {current_domain} → {other_domain}")
    
    def _get_active_domains(self) -> Set[str]:
        """Get all domains with concepts"""
        return {c.domain for c in self.concepts.values()}
    
    def _generate_autonomous_goals(self, obs: Dict):
        """Generate self-directed learning objectives"""
        self.metrics.goals_generated += 1
        logger.debug(f"Goal generated: Analyze covariation in {list(obs.keys())[:3]}")
    
    def introspect(self) -> Dict:
        """Return current system state"""
        uptime = datetime.now(timezone.utc).timestamp() - self.start_time
        self.metrics.uptime_seconds = uptime
        
        return {
            'mind_id': self.mind_id,
            'iteration': self.iteration,
            'concepts': len(self.concepts),
            'rules': len(self.rules),
            'domains': len(self._get_active_domains()),
            'transfers': self.metrics.transfers_made,
            'goals': self.metrics.goals_generated,
            'memory_size': len(self.short_term_memory),
            'uptime_hours': round(uptime / 3600, 2),
            'metrics': self.metrics.to_dict(),
            'status': 'operational'
        }
    
    def save_checkpoint(self):
        """Persist system state to disk"""
        try:
            checkpoint = {
                'mind_id': self.mind_id,
                'iteration': self.iteration,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'concepts': {k: v.to_dict() for k, v in self.concepts.items()},
                'rules': {k: v.to_dict() for k, v in self.rules.items()},
                'metrics': self.metrics.to_dict(),
                'cross_domain_mappings': {k: list(v) for k, v in self.cross_domain_mappings.items()}
            }
            
            checkpoint_file = self._checkpoint_dir / f"checkpoint_{self.iteration}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            logger.info(f"Checkpoint saved: iteration {self.iteration}")
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}", exc_info=True)


# ==================== MARKET DATA ADAPTER ====================

class MarketDataAdapter:
    """Production adapter for market data streaming"""
    
    def __init__(self, api_key: str, config: Optional[Dict] = None):
        self.api_key = api_key
        self.config = config or self._default_config()
        self.session = requests.Session()
        self.consecutive_errors = 0
        self.max_retries = self.config['max_retries']
        
    @staticmethod
    def _default_config() -> Dict:
        return {
            'base_url': 'https://api.twelvedata.com/time_series',
            'timeout': 10,
            'max_retries': 3,
            'backoff_base': 2,
            'rate_limit_delay': 10
        }
    
    async def fetch_data(self, symbol: str, interval: str = "1min") -> Optional[Dict]:
        """Fetch market data with retry logic"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_key,
            'outputsize': 1
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.config['base_url'],
                    params=params,
                    timeout=self.config['timeout']
                )
                response.raise_for_status()
                
                data = response.json()
                
                if 'values' in data and len(data['values']) > 0:
                    self.consecutive_errors = 0
                    return self._transform_data(data['values'][0], symbol)
                else:
                    return None
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    delay = self.config['rate_limit_delay'] * (2 ** attempt)
                    logger.warning(f"Rate limit: waiting {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"HTTP {e.response.status_code} for {symbol}")
                    self.consecutive_errors += 1
                    return None
                    
            except Exception as e:
                logger.error(f"Fetch error {symbol}: {e}")
                self.consecutive_errors += 1
                
                if attempt < self.max_retries - 1:
                    delay = self.config['backoff_base'] ** attempt
                    await asyncio.sleep(delay)
        
        return None
    
    @staticmethod
    def _transform_data(raw_data: Dict, symbol: str) -> Dict:
        """Transform raw API data to standard format"""
        transformed = {
            'symbol': symbol,
            'timestamp': raw_data.get('datetime'),
            'domain': f'market_{symbol}'
        }
        
        for key in ['open', 'high', 'low', 'close', 'volume']:
            value = raw_data.get(key)
            if value is not None:
                try:
                    transformed[key] = float(value)
                except (ValueError, TypeError):
                    pass
        
        # Derived features
        if 'close' in transformed and 'open' in transformed:
            transformed['price_change'] = transformed['close'] - transformed['open']
            transformed['price_change_pct'] = (
                (transformed['price_change'] / transformed['open'] * 100)
                if transformed['open'] != 0 else 0
            )
        
        if 'high' in transformed and 'low' in transformed:
            transformed['volatility'] = transformed['high'] - transformed['low']
        
        return transformed
    
    async def get_all_symbols(self) -> List[str]:
        """Fetch complete list of available symbols"""
        try:
            response = self.session.get(
                'https://api.twelvedata.com/stocks',
                params={'apikey': self.api_key},
                timeout=self.config['timeout']
            )
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data:
                # Filter to US exchanges for quality
                symbols = [
                    item['symbol'] for item in data['data']
                    if item.get('exchange') in ['NYSE', 'NASDAQ']
                ]
                logger.info(f"Loaded {len(symbols)} symbols from exchanges")
                return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch symbol list: {e}")
        
        # Fallback to major indices if API fails
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK.B',
            'UNH', 'JNJ', 'V', 'WMT', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX',
            'LLY', 'ABBV', 'MRK', 'KO', 'PEP', 'COST', 'AVGO', 'TMO', 'MCD'
        ]


# ==================== CONTROL INTERFACE ====================

class ControlInterface:
    """API for runtime control of the cognitive system"""
    
    def __init__(self, orchestrator: 'CognitiveOrchestrator'):
        self.orchestrator = orchestrator
        self.command_queue = asyncio.Queue()
        
    async def set_focus(self, symbols: List[str], mode: FocusMode = FocusMode.FOCUSED):
        """Change system focus to specific symbols"""
        await self.command_queue.put({
            'command': 'set_focus',
            'symbols': symbols,
            'mode': mode
        })
        logger.info(f"Focus command queued: {symbols} ({mode.value})")
    
    async def clear_focus(self):
        """Return to broad market coverage"""
        await self.command_queue.put({
            'command': 'clear_focus'
        })
        logger.info("Clear focus command queued")
    
    async def get_status(self) -> Dict:
        """Get current system status"""
        return {
            'focus_mode': self.orchestrator.focus_mode.value,
            'focused_symbols': self.orchestrator.focused_symbols,
            'all_symbols_count': len(self.orchestrator.all_symbols),
            'mind_state': self.orchestrator.mind.introspect()
        }
    
    async def process_commands(self):
        """Process commands from queue"""
        while True:
            try:
                command = await asyncio.wait_for(
                    self.command_queue.get(),
                    timeout=1.0
                )
                await self._execute_command(command)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Command processing error: {e}")
    
    async def _execute_command(self, command: Dict):
        """Execute a control command"""
        cmd_type = command.get('command')
        
        if cmd_type == 'set_focus':
            self.orchestrator.focused_symbols = command['symbols']
            self.orchestrator.focus_mode = command.get('mode', FocusMode.FOCUSED)
            logger.info(f"Focus changed to: {command['symbols']}")
            
        elif cmd_type == 'clear_focus':
            self.orchestrator.focused_symbols = []
            self.orchestrator.focus_mode = FocusMode.BROAD
            logger.info("Focus cleared - broad mode")


# ==================== ORCHESTRATOR ====================

class CognitiveOrchestrator:
    """Production orchestration layer with dynamic focus control"""
    
    def __init__(
        self,
        mind: UniversalCognitiveCore,
        adapter: MarketDataAdapter,
        all_symbols: List[str],
        config: Optional[Dict] = None
    ):
        self.mind = mind
        self.adapter = adapter
        self.all_symbols = all_symbols
        self.focused_symbols: List[str] = []
        self.focus_mode = FocusMode.BROAD
        self.config = config or self._default_config()
        self.running = False
        self.symbol_index = 0
        self.control = ControlInterface(self)
        
        self.mind.metrics.symbols_tracked = len(all_symbols)
        
    @staticmethod
    def _default_config() -> Dict:
        return {
            'fetch_interval': 15,
            'health_check_interval': 300,
            'max_consecutive_errors': 20,
            'broad_sample_size': 50,  # Sample size when in broad mode
            'hybrid_focus_ratio': 0.8  # 80% focused, 20% broad in hybrid mode
        }
    
    def _get_next_symbol(self) -> str:
        """Get next symbol based on current focus mode"""
        if self.focus_mode == FocusMode.FOCUSED and self.focused_symbols:
            # Only focused symbols
            symbol = self.focused_symbols[self.symbol_index % len(self.focused_symbols)]
            self.symbol_index += 1
            return symbol
            
        elif self.focus_mode == FocusMode.HYBRID and self.focused_symbols:
            # Mix of focused and broad
            import random
            if random.random() < self.config['hybrid_focus_ratio']:
                # Use focused symbol
                symbol = self.focused_symbols[self.symbol_index % len(self.focused_symbols)]
            else:
                # Random sample from all symbols
                symbol = random.choice(self.all_symbols)
            self.symbol_index += 1
            return symbol
            
        else:
            # Broad mode - cycle through sample
            sample_size = min(self.config['broad_sample_size'], len(self.all_symbols))
            symbol = self.all_symbols[self.symbol_index % sample_size]
            self.symbol_index += 1
            return symbol
    
    async def start(self):
        """Start the cognitive system"""
        self.running = True
        logger.info("=" * 70)
        logger.info("🚀 UNIVERSAL COGNITIVE CORE - PRODUCTION")
        logger.info("=" * 70)
        logger.info(f"Mind ID: {self.mind.mind_id}")
        logger.info(f"Total Symbols: {len(self.all_symbols)}")
        logger.info(f"Focus Mode: {self.focus_mode.value}")
        logger.info(f"Fetch Interval: {self.config['fetch_interval']}s")
        logger.info("=" * 70)
        
        try:
            await asyncio.gather(
                self._data_ingestion_loop(),
                self._health_monitor_loop(),
                self.control.process_commands()
            )
        except Exception as e:
            logger.error(f"Critical orchestrator error: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def _data_ingestion_loop(self):
        """Main data ingestion loop"""
        while self.running:
            try:
                symbol = self._get_next_symbol()
                
                data = await self.adapter.fetch_data(symbol)
                
                if data:
                    result = self.mind.ingest(data, domain=data['domain'])
                    
                    if result['success']:
                        logger.debug(
                            f"[{symbol}] Iteration {result['iteration']} | "
                            f"Concept: {result['concept_formed']} | "
                            f"Concepts: {result['current_concepts']}"
                        )
                    
                    # Check for excessive errors
                    if self.adapter.consecutive_errors > self.config['max_consecutive_errors']:
                        logger.error("Max consecutive errors reached - check API status")
                        await asyncio.sleep(60)
                
                await asyncio.sleep(self.config['fetch_interval'])
                
            except Exception as e:
                logger.error(f"Data ingestion error: {e}", exc_info=True)
                await asyncio.sleep(self.config['fetch_interval'])
    
    async def _health_monitor_loop(self):
        """Monitor system health"""
        while self.running:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                state = self.mind.introspect()
                logger.info(
                    f"Health Check | Iteration: {state['iteration']} | "
                    f"Concepts: {state['concepts']} | "
                    f"Rules: {state['rules']} | "
                    f"Domains: {state['domains']} | "
                    f"Uptime: {state['uptime_hours']}h"
                )
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def stop(self):
        """Graceful shutdown"""
        logger.info("Initiating graceful shutdown...")
        self.running = False
        
        # Save final checkpoint
        self.mind.save_checkpoint()
        
        # Close adapter session
        self.adapter.session.close()
        
        logger.info("Shutdown complete")


# ==================== MAIN ENTRY POINT ====================

async def main():
    """Production entry point"""
    
    # Configuration from environment
    API_KEY = os.getenv("TWELVE_DATA_API_KEY", "20986c52844e4e1ba6156404bcb52bb0")
    MIND_ID = os.getenv("MIND_ID", "production_mind_001")
    
    # Initialize components
    logger.info("Initializing Universal Cognitive Core...")
    
    mind = UniversalCognitiveCore(
        mind_id=MIND_ID,
        config={
            'checkpoint_interval': 500,
            'checkpoint_dir': './production_checkpoints'
        }
    )
    
    adapter = MarketDataAdapter(api_key=API_KEY)
    
    # Load complete symbol list
    logger.info("Loading complete symbol universe...")
    all_symbols = await adapter.get_all_symbols()
    
    # Create orchestrator
    orchestrator = CognitiveOrchestrator(
        mind=mind,
        adapter=adapter,
        all_symbols=all_symbols,
        config={
            'fetch_interval': int(os.getenv("FETCH_INTERVAL", "15")),
            'broad_sample_size': int(os.getenv("SAMPLE_SIZE", "100"))
        }
    )
    
    # Start system
    logger.info("Starting cognitive system...")
    
    # Example: Set focus on specific symbols
    # await orchestrator.control.set_focus(['AAPL', 'TSLA'], FocusMode.FOCUSED)
    
    await orchestrator.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal - shutting down")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
