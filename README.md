# ArbRadar v2

Professional prediction market arbitrage tracker. Scans Polymarket, PredictIt, Manifold, Metaculus, and Opinion Markets for cross-platform arbs, dutch-book edges, and tail-end plays. Scores tradeability and sizes bets with Kelly criterion.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Open http://localhost:8000 in your browser.

## API Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Platform health, latency, uptime |
| `GET /markets/polymarket` | Polymarket active markets |
| `GET /markets/predictit` | PredictIt contracts |
| `GET /markets/manifold` | Manifold binary markets |
| `GET /markets/metaculus` | Metaculus forecasts |
| `GET /markets/opinion` | Opinion Markets |
| `GET /markets/all` | All platforms combined with counts and errors |
| `GET /arb/cross?min_gap=2&min_volume=0` | Cross-platform arbitrage |
| `GET /arb/dutch?min_edge=1` | Dutch-book (multi-outcome mispricing) |
| `GET /arb/tail?min_prob=92` | Tail-end (near-certain underpriced) |
| `GET /arb/combinatorial?threshold=15` | Logically related mispricings |
| `GET /arb/all` | All arb types combined |
| `POST /score` | Tradeability scorer (0-100, A-F grade) |
| `GET /market/{platform}/{id}/depth` | Order book depth (Polymarket) |
| `GET /history` | Trade log |
| `POST /history/log` | Log an opportunity |
| `PATCH /history/{id}` | Update logged trade |
| `GET /stats` | Trading statistics |

## Platforms

- **Polymarket** — Free, no API key. USDC on Polygon. 2% fee.
- **PredictIt** — Free, no key. US political markets. 10% profit fee + 5% withdrawal.
- **Manifold** — Free, no key. Play money + real money. 0% fee.
- **Metaculus** — Free, no key. Forecasting community. 0% fee (requires auth as of 2026).
- **Opinion** — Free, international. ~2% fee.

Always calculate fee-adjusted gap before trading. A 3% raw gap on PredictIt becomes -7% after fees.

## Fee Structure

| Platform | Fee | Impact |
|---|---|---|
| Polymarket | 2% on winnings | Subtract 2% from raw gap |
| PredictIt | 10% on profits + 5% withdrawal | Subtract 10% from raw gap |
| Manifold | 0% | No adjustment |
| Metaculus | 0% | No adjustment |
| Opinion | ~2% | Subtract 2% from raw gap |
