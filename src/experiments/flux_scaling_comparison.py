"""
flux_scaling_comparison.py — Log vs Binary vs Linear Flux
==========================================================
The current champion uses:
    flux += log(1 + energy * 100) * 0.5

This was never compared to simpler rules in a controlled experiment.
This script tests three rules head-to-head:

  log     (current) : log(1 + energy*100) * 0.5
  binary            : +1.0 if energy > 0.1 else −0.5 if energy < 0.05
  linear            : energy * 5.0

Live window per mode. Comparison chart at the end.

Usage:
  cd src
  python -m experiments.flux_scaling_comparison                    # 2000 eps, 3 seeds
  python -m experiments.flux_scaling_comparison --episodes 200 --seeds 1
"""

import argparse
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym

from core.steering import SteeringController
from core.tda import TopologicalBrain
from core.agent import QSMA_Agent
from models.bridge import InstinctSeeder, DreamTeacher, GovernorGate
from models.niodoo import NiodooMemory
from experiments.experiment_utils import (
    LiveDashboard, ComparisonDashboard, SeedManager,
    save_json, save_png, rolling_mean, PALETTE, CONFIG_COLORS
)

FLUX_MODES   = ['log', 'binary', 'linear']
TDA_INTERVAL  = 5
DREAM_INTERVAL = 10
SEED_EPISODES  = 100
INSTINCT_WEIGHT = 0.5
DREAM_FRACTION  = 0.3


def _compute_flux_delta(energy: float, mode: str) -> float | None:
    """Return the flux increment to add, or None to skip."""
    if mode == 'log':
        if energy > 0.1:
            return math.log(1 + energy * 100) * 0.5
        elif energy < 0.05:
            return -math.log(1 + (0.05 - energy) * 50) * 0.5
    elif mode == 'binary':
        if energy > 0.1:
            return 1.0
        elif energy < 0.05:
            return -0.5
    elif mode == 'linear':
        if energy > 0.1:
            return energy * 5.0
        elif energy < 0.05:
            return -(0.05 - energy) * 5.0
    return None


def run_flux_mode(mode: str, total_episodes: int, seed: int,
                  dashboard: LiveDashboard) -> dict:

    SeedManager.apply(seed)
    env = gym.make('MountainCar-v0')

    ctrl    = SteeringController()
    brain   = TopologicalBrain(ctrl)
    agent   = QSMA_Agent()
    memory  = NiodooMemory(beta_threshold=0.4)

    seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES,
                            instinct_weight=INSTINCT_WEIGHT)
    trajectories = seeder.generate_seeds(agent)
    teacher = DreamTeacher(trajectories, dream_fraction=DREAM_FRACTION)
    teacher.inject_dreams(agent)
    governor = GovernorGate(initial_threshold=0.8, final_threshold=0.1,
                            warmup_episodes=200, rampdown_episodes=1500)

    success_count = 0
    first_success = None
    win_flags     = []
    flux_snapshots = {}   # episode → max flux value (for heatmap evolution plot)

    for episode in range(total_episodes):
        state, _ = env.reset()
        done  = False
        steps = 0
        max_pos = -1.2
        prev_node_id = None

        phase = 1 if episode < 500 else (2 if episode < 1500 else 3)
        agent.sync(ctrl)
        agent.params['beta'] = max(0.1, 1.5 * (0.995 ** episode))

        while not done and steps < 500:
            agent_action = agent.act(state)

            if phase < 3:
                final_action, _ = governor.gate(state, agent_action, agent, episode, steps)
            else:
                final_action = agent_action

            next_state, reward, terminated, truncated, _ = env.step(final_action)
            done = terminated or truncated

            # ── Custom flux update (bypassing agent.learn flux block) ──
            s  = agent.discretize(state)
            ns = agent.discretize(next_state)
            best_next = 0.0 if done else np.max(agent.q_table[ns])

            phi_now  = math.sin(3 * next_state[0]) + 100 * (next_state[1] ** 2)
            phi_prev = math.sin(3 * state[0])       + 100 * (state[1] ** 2)
            energy_delta = phi_now - phi_prev
            if energy_delta < 0:
                energy_delta *= 1.15

            shaped_reward = reward + energy_delta * 10.0
            agent.q_table[s, final_action] += 0.2 * (
                shaped_reward + 0.999 * best_next - agent.q_table[s, final_action]
            )

            position, velocity = next_state
            energy = (position ** 2) + (velocity ** 2)

            # Apply selected flux rule
            delta = _compute_flux_delta(energy, mode)
            if delta is not None:
                agent.flux[s, final_action] += delta

            agent.flux[s, final_action] *= (1.0 - agent.params['decay'])
            agent.flux[s, final_action] = np.clip(agent.flux[s, final_action], -5.0, 20.0)

            agent._episode_energy_states.append((state.copy(), final_action, energy_delta, reward))

            # Daydream watcher
            if len(agent._episode_energy_states) % agent.daydream_interval == 0:
                agent.watcher.observe(
                    agent._episode_energy_states[-agent.daydream_interval:],
                    agent.flux, agent.splat_memory, agent.discretize
                )

            # TDA
            q   = agent.q_table[s, final_action]
            fl  = agent.flux[s, final_action]
            dlt = abs(reward + 0.999 * np.max(agent.q_table[ns]) - agent.q_table[s, final_action])
            brain.add_log([state[0], state[1], q, fl, dlt, energy])

            agent.replay_buffer.append((state.copy(), final_action, reward,
                                        next_state.copy(), energy))

            content = f"pos:{state[0]:.2f},vel:{state[1]:.3f},a:{final_action},e:{energy:.3f}"
            node_id = memory.flinch_tag(content, state[1], state)
            if prev_node_id:
                memory.connect_nodes(prev_node_id, node_id, weight=abs(state[1]))
            prev_node_id = node_id

            state = next_state
            steps += 1
            max_pos = max(max_pos, state[0])

        win = max_pos >= 0.5
        win_flags.append(1 if win else 0)

        if win:
            success_count += 1
            if first_success is None:
                first_success = episode
            agent.commit_episode(True)
        else:
            agent.commit_episode(False)

        governor.episode_done()
        agent.splat_memory.decay_and_consolidate()
        splat_stats = agent.splat_memory.get_stats()
        dashboard.update(episode, max_pos, steps, agent=agent, splat_stats=splat_stats)

        if (episode + 1) % TDA_INTERVAL == 0 and len(brain.buffer) >= 100:
            brain.analyze()
            ctrl.normalize()

        if (episode + 1) % DREAM_INTERVAL == 0:
            teacher.dream_with_teacher(agent)

        if (episode + 1) % 50 == 0:
            memory.curate_prune()

        if episode in (500, 1000, 1500, total_episodes - 1):
            flux_snapshots[episode] = float(agent.flux.max())

        if (episode + 1) % 200 == 0:
            pct = success_count / (episode + 1) * 100
            print(f'  [flux:{mode}] Ep {episode+1} | Wins: {success_count} ({pct:.1f}%)')

    env.close()

    return {
        'mode': mode, 'seed': seed,
        'total_episodes': total_episodes,
        'wins': success_count,
        'win_pct': success_count / total_episodes * 100,
        'first_success': first_success,
        'win_flags': win_flags,
        'flux_snapshots': flux_snapshots,
    }


def plot_flux_heatmaps(all_results: list, episodes: int):
    """Bonus: for each mode, show flux landscape at ep500, ep1000, ep2000."""
    # We'll use the win_pct as a proxy heatmap since we didn't store full arrays
    # Just show the summary comparison
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seeds',    type=int, default=3)
    args = parser.parse_args()

    seeds = SeedManager.get_seeds(args.seeds)
    comp  = ComparisonDashboard('Flux Scaling: Log vs Binary vs Linear')
    all_results = []

    for mode in FLUX_MODES:
        color = CONFIG_COLORS.get(mode, '#888888')
        print(f'\n{"="*60}')
        print(f'  FLUX MODE: {mode.upper()}')
        print(f'{"="*60}')

        mode_win_flags   = []
        mode_first_success = None

        for seed in seeds:
            print(f'\n  ── Seed {seed} ──')
            dash = LiveDashboard(
                title=f'Flux: {mode} (seed={seed})',
                total_episodes=args.episodes,
                config_color=color
            )
            result = run_flux_mode(mode, args.episodes, seed, dash)
            dash.close(name=f'flux_{mode}_seed{seed}')

            all_results.append(result)
            mode_win_flags.extend(result['win_flags'])
            if result['first_success'] is not None:
                if mode_first_success is None or result['first_success'] < mode_first_success:
                    mode_first_success = result['first_success']

            print(f'  ✓ flux={mode} seed={seed}: '
                  f'{result["wins"]}/{args.episodes} ({result["win_pct"]:.1f}%) '
                  f'first={result["first_success"]}')

        comp.add_result(mode, mode_win_flags, mode_first_success)

    # Summary
    print('\n' + '='*60)
    print('  FLUX SCALING RESULTS')
    print('='*60)
    print(f'  {"Mode":<10} {"Mean Win%":>10} {"±Std":>8} {"First Success":>15}')
    print('-'*60)
    for mode in FLUX_MODES:
        rlist = [r for r in all_results if r['mode'] == mode]
        wpts  = [r['win_pct'] for r in rlist]
        feps  = [r['first_success'] for r in rlist if r['first_success'] is not None]
        avg_f = int(np.mean(feps)) if feps else None
        print(f'  {mode:<10} {np.mean(wpts):>9.1f}% {np.std(wpts):>7.1f}%  {str(avg_f):>15}')
    print('='*60)

    save_json({'episodes': args.episodes, 'seeds': seeds, 'results': all_results}, 'flux_scaling')
    comp.show(name='flux_comparison')


if __name__ == '__main__':
    main()
