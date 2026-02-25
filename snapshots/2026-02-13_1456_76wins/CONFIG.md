# Snapshot: 76 Wins / 1000 Episodes
## Date: 2026-02-13 14:56 PST
## Result: 76/1000 successes (53 in last 200 = 26.5% mastery rate)

---

## Key Parameters

### Agent (agent.py)
- state_bins: 40 (40x40 = 1600 states)
- alpha (learning rate): 0.2
- gamma (discount): 0.999
- initial epsilon: 0.3
- flux cap: 5.0
- flux growth: +0.5 when energy > 0.1
- attractor radius: 0.6
- curiosity boost: 5.0

### Reward Shaping (YIN-YANG)
- Potential function: Φ(s) = sin(3*pos) + 100 * vel²
- Shaped reward: base_reward + (Φ(s') - Φ(s)) * 10.0
- This creates BALANCED positive/negative signals from energy change

### Steering (steering.py)
- initial decay_rate: 0.05
- initial exploration_rate: 0.1
- spike cooldown: 10 TDA intervals
- spike intensity: +0.3 * persistence, capped at 0.5
- normalize decay factor: 0.85
- normalize exploration factor: 0.90
- exploration floor: 0.01

### TDA (tda.py)
- buffer: 2000 max (deque)
- analysis window: last 500 points only
- loop density threshold: 0.3
- H1 persistence threshold: 0.5
- point cloud sample: 200 points
- normalization: mean=0, std=1

### Main Loop (main.py)
- TOTAL_EPISODES: 1000
- TDA_INTERVAL: 5 episodes
- CONFIDENCE SCALING (beta): max(0.1, 1.5 * 0.995^episode)
  - Ep 0: beta = 1.50
  - Ep 100: beta = 0.91
  - Ep 200: beta = 0.55
  - Ep 300: beta = 0.33
  - Ep 400: beta = 0.20
  - Ep 500: beta = 0.12
  - Ep 600+: beta ≈ 0.10 (floor)

---

## Phase Analysis (from this run)

| Phase | Episodes | Wins | AvgPos | Q_max | Flux | Beta | AvgReward |
|-------|----------|------|--------|-------|------|------|-----------|
| Early Habit | 0-200 | 0 | -0.285 | +0.0 | 3.9 | 0.95 | -0.98 |
| Mid Learning | 200-400 | 1 | -0.030 | +0.2 | 3.3 | 0.35 | -0.96 |
| Transition | 400-600 | 3 | +0.046 | +0.4 | 4.3 | 0.13 | -0.95 |
| Breakthrough | 600-800 | 8 | +0.167 | +0.5 | 3.9 | 0.10 | -0.94 |
| Mastery | 800-1000 | 53 | +0.298 | +0.8 | 4.4 | 0.10 | -0.92 |

---

## Key Insights
1. Yin-Yang reward: energy change creates balanced pos/neg signal
2. Confidence scaling: beta fades flux as Q matures
3. Complacency trap at Ep 200-400: agent felt reward from swinging, got stuck
4. Beta hitting 0.1 floor was the breakthrough trigger
5. Spike cooldown prevents runaway TDA over-correction
6. Recent-only TDA analysis (500 pts) prevents old data poisoning
