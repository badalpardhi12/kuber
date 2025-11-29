# Kuber – Research-Grade Market Lab

> High-frequency data ingest, feature engineering, PPO-based reinforcement learning,
> and benchmark tooling for repeatable trading research.

---

## Table of Contents
- [Disclaimer](#⚠️-disclaimer)
- [Highlights](#highlights)
- [System Overview](#system-overview)
- [Getting Started](#getting-started)
- [Training SOTA RL Agents](#training-sota-rl-agents)
- [Benchmarking](#benchmarking)
- [Repository Layout](#repository-layout)
- [Legacy Components](#legacy-components)
- [Troubleshooting & Tips](#troubleshooting--tips)
- [Support](#support)

---

## ⚠️ Disclaimer
- **Financial risk:** Models are research artifacts. Markets change, capital can be lost. Use at your own risk.
- **Visa compliance:** The automation is rule-based and intended for passive execution. Confirm the regulatory implications of running active trading systems in your jurisdiction.

---

## Highlights
- **Intraday data parity** – Polygon.io 5-minute history with automatic Yahoo fallback keeps the training set (last 2 years) separate from the test set (most recent 60 days).
- **Shared feature pipeline** – `kuber/data/features.py` adds momentum, volatility, and volume-normalized signals once and shares them between training and benchmarking.
- **Gymnasium trading env** – `kuber/rl/envs/trading_env.py` tracks mark-to-market equity, commissions, and exposes deterministic sliding-window observations.
- **Transformer-enhanced PPO** – `scripts/train_rl_agent.py` wires Stable-Baselines3 PPO to a custom transformer feature extractor, observation/reward normalization, learning-rate schedules, KL targeting, and evaluation callbacks that checkpoint the best model.
- **VectorBT benchmark** – `tests/benchmark_strategies.py` rebuilds positions directly from PPO actions to perform fair comparisons against Buy & Hold, RSI, and Golden Cross.
- **Classical stack still available** – Legacy broker, engine, dashboard, and rule-based strategies remain intact for Robinhood automation or hybrid research.

---

## System Overview
**Data** → `kuber/data/providers.py` downloads Polygon intraday candles, caches them, and gracefully falls back to Yahoo when the API key is missing or throttled.

**Feature Engineering** → `kuber/data/features.py` adds SMA20, EMA50, MACD, RSI, stochastic, ATR, Bollinger width, log returns, and normalized volume ratios. The identical transformation is applied in both train and inference paths.

**RL Environment** → `kuber/rl/envs/trading_env.py` exposes a continuous action space (target net exposure in [-1, 1]). Each step updates equity, subtracts commissions, and emits the last 30-bar feature tensor as the observation. Original timestamps are preserved for downstream evaluation alignment.

**Training Script** → `scripts/train_rl_agent.py` handles:
- Parallel ticker training (CPU-count aware) with 1M timesteps default.
- Transformer feature extractor (3 encoder blocks, 8 heads, SiLU activations) before PPO heads.
- Linear schedules for LR and clip range, 4096-step rollouts, 512 batch size, 20 epochs, KL targeting, entropy encouragement, and VecNormalize.
- Holdout eval env + `StopTrainingOnNoModelImprovement` callback. Best checkpoints are copied under `models/best_models/<ticker>/` and normalization statistics are persisted as `vec_normalize_<ticker>.pkl`.
- Opt-in TensorBoard logging via `KUBER_ENABLE_TENSORBOARD=1`.

**Benchmark** → `tests/benchmark_strategies.py` fetches the most recent 60 days of 5-minute data, recomputes features, replays PPO policies through the gym env, and translates each action into a portfolio weight. VectorBT computes return, Sharpe, drawdown, and win rate alongside classical baselines.

---

## Getting Started
### Prerequisites
- macOS/Linux/WSL with Python 3.11 recommended (3.9+ supported).
- Polygon.io API key for full intraday history. Without it, yfinance will deliver hourly data and training/testing periods may overlap.
- Optional: Robinhood credentials if you intend to experiment with the legacy live engine.

### Install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file (copy `.env.example`) and set the following:
```env
POLYGON_API_KEY=your_polygon_key_here  # required for full 5m data
ROBINHOOD_USERNAME=optional
ROBINHOOD_PASSWORD=optional
ROBINHOOD_TOTP_SECRET=optional
```
Runtime overrides supported by the training script:
- `KUBER_PPO_TIMESTEPS` – default `1_000_000`. Lower for experiments.
- `KUBER_ENABLE_TENSORBOARD=1` – enable TB logging after installing `pip install tensorboard`.

---

## Training SOTA RL Agents
```bash
source .venv/bin/activate
python scripts/train_rl_agent.py
```
What happens:
1. Each ticker downloads two years of 5-minute bars (ending 60 days ago) for training and preserves the most recent 60 days for evaluation only.
2. Feature engineering runs once per dataset.
3. SB3 PPO + transformer trains for the configured steps and periodically evaluates on the holdout window.
4. Models and VecNormalize stats are stored under `models/`.

Artifacts:
```
models/
├── ppo_agent_SPY.zip
├── vec_normalize_SPY.pkl
├── best_models/SPY/best_model.zip
└── eval_logs/SPY/evaluations.npz
```
Inspect evaluation logs:
```bash
python - <<'PY'
import numpy as np, pathlib
for sym in sorted(pathlib.Path('models/eval_logs').glob('*')):
    pk = sym / 'evaluations.npz'
    if pk.exists():
        data = np.load(pk)
        best = data['results'].mean(axis=1).max()
        print(f"{sym.name}: best eval reward {best:.2f}")
PY
```
Pause/resume: because each ticker trains in its own process, re-running the script will simply retrain from scratch. Kill specific processes if you need to stop early.

---

## Benchmarking
```bash
source .venv/bin/activate
python tests/benchmark_strategies.py
```
The benchmark fetches the most recent 60 trading days (5-minute bars) and reports:

| Strategy              | Total Return | Sharpe | Max Drawdown | Win Rate |
|-----------------------|-------------:|-------:|-------------:|---------:|
| Buy & Hold            |       ~1.7 % |  0.17  |     -13.9 %  | 100 %    |
| RSI 30/70             |     -32.5 %  | -16.4  |     -34.0 %  | 27.8 %   |
| Golden Cross 50/200   |      -4.9 %  | -3.0   |     -12.0 %  | 28.3 %   |
| **SOTA RL (PPO)**     |       ~2.6 % | -0.06  |      14.1 %  | 48.1 %   |

Numbers will shift as models improve; re-run the training script for longer horizons or tuned hyperparameters.

---

## Repository Layout
```
kuber/
├── data/             # providers + shared indicators
├── rl/
│   ├── envs/         # gym-compatible trading env
│   ├── policies.py   # transformer feature extractor
│   └── ...
├── strategies/       # classical TA strategies
├── core/             # Robinhood broker & trading engine
├── dashboard/        # Streamlit UI
├── scripts/          # training utilities (PPO, transformers)
├── tests/            # vectorbt benchmark and regression tests
└── models/           # generated artifacts (ignored by Git)
```

---

## Legacy Components
The Robinhood broker, streaming engine, risk module, and Streamlit dashboard remain available for experimentation, but they are not wired into the new PPO pipeline yet. Use them for paper trading or to prototype discretionary logic while the RL stack matures.

---

## Troubleshooting & Tips
- **Polygon throttling:** When the API issues 429s the script automatically falls back to yfinance hourly bars. Re-run later for the full 5-minute fidelity.
- **TensorBoard:** Disabled by default. Install `tensorboard` and export `KUBER_ENABLE_TENSORBOARD=1` to capture logs under `tensorboard/<ticker>/`.
- **Hardware:** PPO rollouts are CPU-bound; MPS support on Apple Silicon is enabled but yields similar throughput to CPU for MLP/transformer policies.
- **Reward scale:** The environment multiplies portfolio returns by 100 for stability. When analyzing logs, divide reward values by 100 to interpret as percent equity change per step.
- **Extending features:** Add indicators in `kuber/data/features.py` and retrain; the inference path automatically benefits.

---

## Support
- Issues & ideas: open a GitHub issue.
- Email: support@kuber.dev
- Discord: https://discord.gg/kuber (community chat)

Happy researching!
