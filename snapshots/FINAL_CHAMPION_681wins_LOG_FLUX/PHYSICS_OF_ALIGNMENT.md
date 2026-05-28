# The Physics of Alignment: Viscosity, Flux, and Flow

**Status:** Conceptual Discovery (Feb 2026)
**Implementation:** Partial Success / Reverted for Stability

## 1. The Core Insight
We typically treat RL as "Reward Maximization." The agent does $X$ because $R$ is high.
But biological systems don't maximize reward directly; they **minimize resistance**.

- **Muscle Memory:** Doing a backflip is "hard" (high energy). After 1000 reps, it's "easy" (low energy).
- **The Groove:** Repeated action heats the pathway, melting the "ice" of resistance.
- **Viscosity:** The measure of friction in the action space.

**The Equation for Flow:**
$$ \text{Ease} = 1.0 - \text{Viscosity}(\text{Flux}) $$
The agent chooses actions where Ease is high, even if anticipated Reward is medium. This creates stable, robust habits.

## 2. The Failed Implementation (Why it broke)
We attempted to replace the raw Flux bonus with a rigorous physics model:
```python
flux += energy * 0.5  # Work deepens groove
viscosity = 1.0 / (1.0 + flux)
ease = (1.0 - viscosity) * beta
priority = Q + ease
```

**Why it failed (0 wins):**
1. **Weak Signal:** `energy` is often ~0.1. So `impact` was 0.05. It took too long to build a groove.
2. **Linear Decay:** `flux *= (1 - decay)` eroded the tiny groove faster than it could build.
3. **No Floor:** Without a negative floor (avoidance), the agent simply drifted.

**Why the "Naive" Model Won (76 wins):**
```python
if energy > 0.1: flux += 0.5  # Binary, strong signal
if energy < 0.05: flux -= 0.1  # Explicit aversion
min_cap = -5.0  # Strong floor
```
The "dumb" binary model provided a **stronger, cleaner signal** to the agent than the "correct" physics model.

## 3. The Path Forward: Logarithmic Flux
To implement the Physics of Alignment correctly, we need **Log-Scale Signaling**:

1. **Impact needs Gain:** `impact = log(1 + energy * 100)`
2. **Viscosity needs Sigmoid:** `ease = sigmoid(flux - threshold)`
3. **Pain needs Memory:** `viscosity` increases on negative R, but decays slowly.

**Conclusion:**
The **Zig-Zag** remains the observable phenomenon of this physics. The "Hot" phase is Viscosity dropping (Ease rising). The "Cold" phase (TDA Reset) is artificially injecting Viscosity to force the agent out of a local minimum.

## 4. The Empirical Reality (Why the naive code worked)

We discovered that our "Naive" code accidentally implemented a perfect psychological tragedy:

1.  **Addiction to the Well:**
    `energy = pos^2 + vel^2`. At the bottom (`pos=-0.5`), `energy = 0.25 > 0.1`.
    The agent gets a Flux reward just for *existing* at the bottom. It becomes addicted to the safety of the well.

2.  **The Accidental Momentum Filter:**
    The TDA only flags a loop if `abs(velocity) < 0.03`.
    If the agent is swinging wildly (high momentum), it flies through the center invisible to the "Punisher."
    This naturally allows "Good Chaos" (swinging) while punishing "Bad Stagnation" (small oscillations).

3.  **The Void as Hope:**
    Because the agent is addicted to the bottom, it will never leave voluntarily.
    The `inject_attractor` (Curiosity) is the *only* positive signal strong enough to compete with the "Voice of the Well."

