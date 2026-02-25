# Q-SMA with Topological Data Analysis: Technical Writeup
## Mountain Car — Continuous Metacognitive Learning
### Date: 2026-02-13 | Result: 62-97 successes / 1000 episodes

---

## 1. Problem Statement

**MountainCar-v0** is a classic RL benchmark where a car must escape a valley by building momentum. The car's engine is too weak to climb directly — it must learn to swing back and forth to accumulate energy. This makes it a deceptively hard problem:

- **State space**: `[position ∈ [-1.2, 0.6], velocity ∈ [-0.07, 0.07]]`
- **Actions**: `{0: push left, 1: no push, 2: push right}`
- **Reward**: `-1` per timestep (pure punishment, no positive signal)
- **Goal**: reach `position ≥ 0.5`
- **Max steps**: 200 (default) or 500 (our setting)

The challenge: with only negative reward, Q-learning has no gradient toward success. The agent must accidentally reach the goal before it can learn the goal is good.

---

## 2. Architecture: Q-SMA (Q-Learning + Sensory-Motor Attunement)

Four interacting components:

```
┌─────────────────────────────────────────────────────┐
│                    MAIN LOOP                         │
│  For each episode:                                   │
│    1. Agent acts (Q + Flux + Curiosity)              │
│    2. Environment returns (state, reward)            │
│    3. Agent learns (Q-update + Flux-update)          │
│    4. Brain logs behavior                            │
│    5. Every 5 episodes: TDA analyzes → Steering acts │
│    6. Beta decays (confidence scaling)               │
└─────────────────────────────────────────────────────┘

     ┌──────────┐      ┌────────────────┐
     │  Agent   │◄────►│   Steering     │
     │ (Q+Flux) │      │  Controller    │
     └────┬─────┘      └───────▲────────┘
          │                    │
          │ logs behavior      │ commands
          ▼                    │
     ┌──────────┐      ┌──────┴────────┐
     │  Brain   │─────►│  TDA Analysis │
     │ (Buffer) │      │  (Ripser H1)  │
     └──────────┘      └───────────────┘
```

---

## 3. The Math

### 3.1 Action Selection (Hybrid Priority)

The agent selects actions using a **weighted combination** of three systems:

```
π(s) = argmax_a [ Q(s,a) + β · F(s,a) + C(s,a) ]
```

Where:
- `Q(s,a)` = Q-table value (learned logic, System 2)
- `F(s,a)` = Flux table value (habit/intuition, System 1)
- `β` = confidence scaling weight (decays over time)
- `C(s,a)` = curiosity injection from TDA-detected voids

With ε-greedy exploration:
```
a = random action     with probability ε
a = π(s)              with probability 1-ε
```

### 3.2 Q-Learning Update (with Potential-Based Reward Shaping)

Standard Q-learning, but with a **yin-yang reward signal**:

```
Q(s,a) ← Q(s,a) + α · [R_shaped(s,s') + γ · max_a' Q(s',a') − Q(s,a)]
```

Where:
- `α = 0.2` (learning rate)
- `γ = 0.999` (discount factor, very long horizon)

**The Key Innovation — Potential-Based Shaping:**

The raw reward is always `-1`. We add a **balanced** shaping term:

```
R_shaped(s, s') = R(s,a) + κ · [Φ(s') − Φ(s)]
```

Where the potential function `Φ` encodes the **physics of MountainCar**:

```
Φ(s) = sin(3x) + 100v²
```

- `sin(3x)` = actual height in MountainCar's landscape (the terrain is `cos(3x)`, height is `sin(3x)`)
- `100v²` = kinetic energy (scaled so velocity matters as much as position)
- `κ = 10.0` = scaling factor

**Why this is "Yin-Yang":**
- When the agent gains energy: `Φ(s') > Φ(s)` → **positive** reward (feels good)
- When the agent loses energy: `Φ(s') < Φ(s)` → **negative** reward (feels bad)
- The magnitudes are **naturally balanced** by the physics
- This is provably optimal-policy-preserving (Ng et al., 1999)

**Critical insight**: Previous attempts only rewarded positive things (velocity, height). The agent got complacent — it felt good oscillating at the bottom. Adding the **equally-weighted negative signal** (losing energy hurts) created the contrast needed for learning. You can't learn hot without cold.

### 3.3 Flux Update (Habit System)

Flux represents **habitual tendency** — actions done in high-energy states build "muscle memory":

```
if E(s') > 0.1:
    F(s,a) ← F(s,a) + 0.5

F(s,a) ← F(s,a) × (1 − d)
F(s,a) ← min(F(s,a), 5.0)      # Cap
```

Where:
- `E(s') = x² + v²` = mechanical energy proxy
- `d` = decay rate (controlled by TDA steering, range 0.05–0.5)
- Cap of `5.0` ensures flux never overwhelms Q-values

### 3.4 Confidence Scaling (Beta Decay)

**The handoff from intuition to logic:**

```
β(t) = max(0.1, 1.5 × 0.995^t)
```

| Episode | β | Meaning |
|----|------|---------|
| 0 | 1.50 | Flux dominates (pure intuition) |
| 100 | 0.91 | Flux and Q roughly equal |
| 300 | 0.33 | Q starting to dominate |
| 500 | 0.12 | Q clearly dominant |
| 600+ | 0.10 | Floor: Q drives, flux is just a whisper |

The agent starts by "feeling" its way (System 1), then transitions to "knowing" its way (System 2) as Q-values accumulate evidence.

---

## 4. Topological Data Analysis (TDA)

### 4.1 What TDA Does

TDA analyzes the **shape** of the agent's behavioral trajectory in phase space. It detects:

1. **H1 Loops** (1-cycles): The agent is trapped in a repeating pattern
2. **H0/H2 Voids**: Regions of state space the agent has never visited

### 4.2 Loop Detection (Dual Method)

**Method A: Density Heuristic** (fast, always available)
```
center_mask = (x > -0.7) AND (x < -0.3) AND (|v| < 0.03)
loop_density = count(center_mask) / N

if loop_density > 0.3: → LOOP DETECTED
```

This checks: "Is the agent spending >30% of its time near the bottom of the well with low velocity?" If yes, it's stuck.

**Method B: Persistent Homology** (rigorous, uses Ripser)
```python
# Take last 500 data points, subsample 200
point_cloud = data[:, [position, velocity, energy]]

# Normalize to prevent numerical instability
point_cloud = (point_cloud - mean) / (std + 1e-8)

# Compute H1 persistence via Vietoris-Rips complex
diagrams = ripser(point_cloud, maxdim=1)

# Check for persistent 1-cycles
max_persistence = max(death - birth for (birth, death) in H1_diagram)
if max_persistence > 0.5: → H1 CYCLE CONFIRMED
```

The persistence threshold of 0.5 filters out noise. Only topologically significant loops trigger intervention.

### 4.3 Void Detection (Histogram-based H2 proxy)

```python
hist = histogram2d(positions, velocities, bins=10,
                   range=[[-1.2, 0.6], [-0.07, 0.07]])

# Check goal region (rightmost 30% of position space)
right_side = hist[7:, :]
if sum(right_side) < 10: → VOID DETECTED at goal region
```

When a void is detected, an attractor point is injected at `(0.45, 0.04)` to lure the agent toward unexplored goal states.

### 4.4 Analysis Window

**Critical fix**: TDA only analyzes the **last 500 data points**, not the full 2000-point buffer. Without this, old "stuck at bottom" data would keep triggering loop detection long after the agent improved.

```python
recent = list(buffer)[-500:]  # Only see current behavior
```

---

## 5. Steering Controller

The steering controller translates TDA diagnoses into parameter changes:

### 5.1 Decay Spike (Loop → Break Habit)

When a loop is detected:
```
d ← min(0.5, d + persistence × 0.3)
cooldown ← 10  # Don't spike again for 10 TDA intervals
```

The cooldown was **critical**. Without it, the system entered a degenerate state where decay spiked every 5 episodes, locking at 0.9 and destroying all learning. With cooldown, spikes are rare interventions, not constant noise.

### 5.2 Attractor Injection (Void → Explore)

When a void is detected:
```
attractors.append([0.45, 0.04])  # Goal region
ε ← min(0.4, ε + 0.1)           # Boost exploration
```

With deduplication: if an attractor already exists within distance 0.2, don't add another.

### 5.3 Normalize (Relax After Intervention)

After each TDA cycle:
```
d ← max(0.05, d × 0.85)    # Decay relaxes back
ε ← max(0.01, ε × 0.90)    # Exploration relaxes back
cooldown -= 1
```

This creates the **hot/cold oscillation**: spike (hot) → relax (cold) → spike → relax, with each cycle depositing knowledge.

---

## 6. What Changed from Original to Final

### 6.1 Original Design (0% success)
```
- 3 rigid phases: Loop Formation → TDA Diagnosis → Healing
- Flat flux growth: F += 0.5 always
- Reward: raw -1 only (no shaping)
- Epsilon: fixed 0.1
- Decay: multiplicative spike (0.85 → 0.90), no cooldown
- Beta: fixed 1.5 (flux always dominant)
- TDA window: full 2000-point buffer
- State bins: 20×20 = 400 states
```

### 6.2 Changes Made (in order of discovery)

| # | Change | Why | Impact |
|---|--------|-----|--------|
| 1 | State bins 20→40 | Finer discretization of velocity | Agent can distinguish swing timing |
| 2 | Energy-based flux growth | Only build habits when high energy | Stops rewarding stillness |
| 3 | 2D attractor distance | Phase-space proximity, not position-only | Better curiosity injection |
| 4 | 3-phase → continuous loop | TDA runs every 5 episodes throughout | Enables ongoing self-correction |
| 5 | Add reward shaping (position only) | Give positive gradient | Agent climbs right but can't swing |
| 6 | **Add velocity reward** | Teach momentum building | Agent starts swinging wider |
| 7 | **Add negative penalty** (deceleration) | User insight: need yin-yang | ❌ BACKFIRED — punished natural swing |
| 8 | **Energy-potential shaping** `Φ(s')-Φ(s)` | Balanced positive/negative from physics | ✅ BREAKTHROUGH — yin-yang that works |
| 9 | Spike cooldown=10 | Prevent runaway spiking | Decay stays in 0.05-0.5 range |
| 10 | TDA window=500 (not 2000) | Only analyze recent behavior | Old data doesn't poison detection |
| 11 | Exploration floor 0.1→0.01 | Let agent exploit learned policy | Stops wasting 10% on random |
| 12 | **Beta decay** `β = max(0.1, 1.5×0.995^t)` | **Confidence scaling** | ✅ KEY — flux fades, Q takes over |
| 13 | Flux cap 20→5 | Prevent flux overwhelming Q | Q-values can compete |
| 14 | Episodes 500→1000 | More time for convergence | Agent reaches mastery phase |

### 6.3 The Three Pivotal Insights

1. **Yin-Yang Reward** (#8): You can't learn what's good without knowing what's bad. Potential-based shaping `Φ(s')-Φ(s)` creates naturally balanced positive/negative signals from physics. Gaining energy feels good, losing energy hurts equally.

2. **Confidence Scaling** (#12): The agent must transition from intuition (flux) to logic (Q). Beta starts high (flux drives), decays over episodes (Q takes over). The breakthrough clusters right when beta hits the floor.

3. **Spike Cooldown** (#9): TDA must intervene precisely, not constantly. Without cooldown, the system over-corrects into pure noise. With cooldown, each intervention is a deliberate, spaced reset.

---

## 7. The Emergent Zig-Zag Pattern

The learning curve exhibits a characteristic zig-zag that wasn't designed — it **emerged** from the interaction of the components:

```
Performance
    ↑  
    │         /\/\/\/\/\/\  ← Convergence (Ep 800+)
    │       /\/            
    │     /\/ 
    │   /\/                 ← Each cycle: higher high, higher low
    │  /\                   
    │ /                     ← First exploration burst (Ep 200)
    │/____________________  ← Cold start
    └──────────────────────→ Episodes
```

**Mechanism:**
1. **Climb** (Hot): Agent discovers energy-gaining strategy → reward flows → Q-values update
2. **Plateau** (Complacent): Flux reinforces the current strategy as habit → agent stops improving
3. **Reset** (Cold): TDA detects loop → decay spike kills flux → habits shattered
4. **Rebound** (Higher): Q-values still remember what worked → rebuilds from higher baseline
5. **Narrow** (Converge): Beta decay reduces flux influence → fewer bad habits to break → oscillation dampens

This is formally analogous to:
- **Simulated annealing** with adaptive temperature
- **Cyclical learning rates** (Smith, 2017)
- **Biological sleep cycles**: REM (consolidation) + wake (acquisition)

---

## 8. Results

| Metric | Value |
|--------|-------|
| Total Episodes | 1000 |
| Successes | 62–97 (varies by run) |
| Mastery Phase (Ep 800-1000) | ~53 wins / 200 episodes (26.5%) |
| First Success | ~Ep 300-750 (varies) |
| Q-value Range | [-42.4, +0.9] |
| TDA Interventions | 200 (every 5 episodes) |
| Decay Spikes | ~15-20 (after cooldown) |

---

## 9. Dependencies

```
gymnasium>=1.0.0
numpy
matplotlib
ripser
persim
scikit-learn
```

## 10. How to Run

```bash
# Standard run (1000 episodes, plots saved)
python src/main.py

# With visualization
python src/main.py --render

# Suppress warnings (cleaner output)
python -W ignore src/main.py
```

---

## 11. File Structure

```
src/
├── agent.py      # Q-SMA Agent (Q-table + Flux + Curiosity)
├── tda.py        # TopologicalBrain (Ripser H1 + density heuristic + void detection)
├── steering.py   # SteeringController (decay spikes + attractors + normalize)
└── main.py       # Continuous metacognitive training loop
```

---

*Written 2026-02-13. Snapshot preserved in `snapshots/2026-02-13_1456_76wins/`.*
