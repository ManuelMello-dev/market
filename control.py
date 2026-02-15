#!/usr/bin/env python3
"""
Control Interface for Universal Cognitive Core
Send commands to running system to change focus, get status, etc.
"""

import asyncio
import json
import sys
from typing import List

# This would connect to your running system via IPC/Redis/HTTP
# For now, it demonstrates the control patterns

class RemoteControl:
    """Remote control interface for cognitive system"""
    
    def __init__(self, system_url: str = "localhost:8080"):
        self.system_url = system_url
    
    async def focus_on(self, symbols: List[str], mode: str = "focused"):
        """Direct system to focus on specific symbols"""
        command = {
            'action': 'set_focus',
            'symbols': symbols,
            'mode': mode
        }
        print(f"Setting focus to: {', '.join(symbols)} ({mode} mode)")
        # Would send to running system
        return command
    
    async def broad_scan(self):
        """Return system to broad market scanning"""
        command = {'action': 'clear_focus'}
        print("Returning to broad market scan")
        return command
    
    async def hybrid_mode(self, primary_symbols: List[str]):
        """Set hybrid mode: focused + periodic broad sampling"""
        command = {
            'action': 'set_focus',
            'symbols': primary_symbols,
            'mode': 'hybrid'
        }
        print(f"Hybrid mode: Focusing on {', '.join(primary_symbols)} + broad sampling")
        return command
    
    async def get_status(self):
        """Query current system status"""
        print("Querying system status...")
        # Would query running system
        status = {
            'focus_mode': 'broad',
            'focused_symbols': [],
            'total_symbols': 5000,
            'iteration': 1234,
            'concepts': 89,
            'domains': 15
        }
        return status
    
    async def print_status(self):
        """Print formatted status"""
        status = await self.get_status()
        print("\n" + "=" * 60)
        print("SYSTEM STATUS")
        print("=" * 60)
        print(f"Mode:           {status['focus_mode'].upper()}")
        print(f"Focused On:     {', '.join(status['focused_symbols']) if status['focused_symbols'] else 'N/A'}")
        print(f"Total Symbols:  {status['total_symbols']}")
        print(f"Iteration:      {status['iteration']}")
        print(f"Concepts:       {status['concepts']}")
        print(f"Domains:        {status['domains']}")
        print("=" * 60 + "\n")


async def main():
    """Example usage"""
    control = RemoteControl()
    
    if len(sys.argv) < 2:
        print("Universal Cognitive Core - Control Interface")
        print("\nUsage:")
        print("  python control.py status                    - Get system status")
        print("  python control.py focus AAPL TSLA          - Focus on specific symbols")
        print("  python control.py hybrid AAPL MSFT GOOGL   - Hybrid mode")
        print("  python control.py broad                     - Broad market scan")
        print("\nExamples:")
        print("  # Focus intensely on Apple and Tesla")
        print("  python control.py focus AAPL TSLA")
        print()
        print("  # Monitor FAANG stocks + sample broader market")
        print("  python control.py hybrid AAPL MSFT GOOGL AMZN META")
        print()
        print("  # Scan entire market")
        print("  python control.py broad")
        return
    
    action = sys.argv[1].lower()
    
    if action == 'status':
        await control.print_status()
    
    elif action == 'focus':
        if len(sys.argv) < 3:
            print("Error: Specify symbols to focus on")
            print("Example: python control.py focus AAPL TSLA NVDA")
            return
        symbols = sys.argv[2:]
        await control.focus_on(symbols, mode='focused')
        print(f"\n✅ System will now focus exclusively on: {', '.join(symbols)}")
    
    elif action == 'hybrid':
        if len(sys.argv) < 3:
            print("Error: Specify primary symbols for hybrid mode")
            print("Example: python control.py hybrid AAPL MSFT GOOGL")
            return
        symbols = sys.argv[2:]
        await control.hybrid_mode(symbols)
        print(f"\n✅ Hybrid mode: 80% focus on {', '.join(symbols)}, 20% broad sampling")
    
    elif action == 'broad':
        await control.broad_scan()
        print("\n✅ System returning to broad market scan mode")
    
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: status, focus, hybrid, broad")


if __name__ == "__main__":
    asyncio.run(main())
