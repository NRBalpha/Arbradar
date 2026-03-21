#!/bin/bash
echo "=== ArbRadar Health Check ==="
echo "Time: $(date)"
echo ""
echo "--- Platform Status ---"
curl -s http://localhost:8000/health | python3 -m json.tool
echo ""
echo "--- Arb Count ---"
curl -s "http://localhost:8000/arb/cross?min_gap=1.5" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Arbs found: {len(d.get(\"opportunities\", []))}')"
echo ""
echo "--- Market Count ---"
curl -s http://localhost:8000/markets/all | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Total markets: {len(d.get(\"markets\", []))}')"
