# Universal Cognitive Market Wanderer

An autonomous cognitive system that continuously observes financial markets, forms concepts, infers rules, and generates autonomous goals. The system demonstrates adaptive learning across market data.

## Features

- **Concept Formation**: Automatically discovers patterns in market data
- **Rule Inference**: Learns relationships between different market metrics
- **Cross-Domain Transfer**: Maps knowledge across different domains
- **Autonomous Goals**: Self-generates research objectives
- **Multi-Source Data**: Uses yfinance and Polygon.io APIs for market data

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python universal_mind.py
```

The system will begin wandering through major tech stocks and cryptocurrencies, forming concepts and learning patterns.

## Configuration

- **POLYGON_API_KEY**: Get a free key at polygon.io
- **delay_seconds**: Time between observations (default: 18 seconds)
- **max_iterations**: Number of observations to make (default: 50)

## How It Works

1. Fetches real-time market data for random symbols
2. Transforms raw data into cognitive observations
3. Forms concepts based on data signatures
4. Infers rules from numeric relationships
5. Attempts cross-domain knowledge transfer
6. Generates autonomous research goals