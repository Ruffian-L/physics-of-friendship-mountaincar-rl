"""
dream_ratio_sweep.py — Optimal Teacher/Own-Experience Ratio
============================================================
The DreamTeacher mixes physics expert trajectories with the agent's own
experience during dream cycles. The current ratio (30% teacher) was
never optimized — it was chosen by intuition.

This script sweeps DREAM_FRACTION over 6 values:
  0.0   — pure own experience (no teacher)
  0.15  — slight guidance
  0.30  — current champion setting
  0.50  — 50/50 split
  0.70  — heavily teacher-guided
  1.0   — pure teacher data (agent never dreams alone)

Usage:
  cd src
  python -m experiments.dream_ratio_sweep                    # 2000 eps, 3 seeds
  python -m experiments.dream_ratio_sweep --episodes 200 --seeds 1
"""

import argparse
import sys, os
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
    save_json, CONFIG_COLORS
)

DREAM_FRACTIONS = [0.0, 0.15, 0.30, 0.50, 0.70, 1.0]
TDA_INTERVAL    = 5
DREAM_INTERVAL  = 10
SEED_EPISODES   = 100
INSTINCT_WEIGHT = 0.5


def run_ratio(dream_fraction: float, total_episodes: int, seed: int,
              dashboard: LiveDashboard) -> dict:

    SeedManager.apply(seed)
    env = gym.make('MountainCar-v0')

    ctrl   = SteeringController()
    brain  = TopologicalBrain(ctrl)
    agent  = QSMA_Agent()
    memory = NiodooMemory(beta_threshold=0.4)

    seeder = InstinctSeeder(num_seed_episodes=SEED_EPISODES,
                            instinct_weight=INSTINCT_WEIGHT)
    trajectories = seeder.generate_seeds(agent)
    teacher = DreamTeacher(trajectories, dream_fraction=dream_fraction)
    teacher.inject_dreams(agent)
    governor = GovernorGate(initial_threshold=0.8, final_threshold=0.1,
                            warmup_episodes=200, rampdown_episodes=1500)

    success_count = 0
    first_success = None
    win_flags     = []

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

            energy = agent.learn(state, final_action, reward, next_state, done=done)

            s   = agent.discretize(state)
            ns  = agent.discretize(next_state)
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
            if dream_fraction == 0.0:
                agent.dream_cycle()       # pure own experience
            else:
                teacher.dream_with_teacher(agent)

        if (episode + 1) % 50 == 0:
            memory.curate_prune()

        if (episode + 1) % 200 == 0:
            pct = success_count / (episode + 1) * 100
            print(f'  [ratio:{dream_fraction}] Ep {episode+1} | '
                  f'Wins: {success_count} ({pct:.1f}%)')

    env.close()
    return {
        'dream_fraction': dream_fraction,
        'seed': seed,
        'total_episodes': total_episodes,
        'wins': success_count,
        'win_pct': success_count / total_episodes * 100,
        'first_success': first_success,
        'win_flags': win_flags,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seeds',    type=int, default=3)
    args = parser.parse_args()

    seeds = SeedManager.get_seeds(args.seeds)
    comp  = ComparisonDashboard('Dream Teacher Ratio Sweep (0% → 100%)')
    all_results = []

    for fraction in DREAM_FRACTIONS:
        cfg_name = str(fraction)
        color    = CONFIG_COLORS.get(cfg_name, '#888888')

        print(f'\n{"="*60}')
        print(f'  DREAM FRACTION: {fraction:.0%} teacher')
        print(f'{"="*60}')

        frac_win_flags   = []
        frac_first_success = None

        for seed in seeds:
            print(f'\n  ── Seed {seed} ──')
            dash = LiveDashboard(
                title=f'Dream Ratio: {fraction:.0%} teacher (seed={seed})',
                total_episodes=args.episodes,
                config_color=color
            )
            result = run_ratio(fraction, args.episodes, seed, dash)
            dash.close(name=f'dream_ratio_{fraction}_seed{seed}')

            all_results.append(result)
            frac_win_flags.extend(result['win_flags'])
            if result['first_success'] is not None:
                if frac_first_success is None or result['first_success'] < frac_first_success:
                    frac_first_success = result['first_success']

            print(f'  ✓ ratio={fraction} seed={seed}: '
                  f'{result["wins"]}/{args.episodes} ({result["win_pct"]:.1f}%) '
                  f'first={result["first_success"]}')

        comp.add_result(cfg_name, frac_win_flags, frac_first_success)

    # Summary
    print('\n' + '='*60)
    print('  DREAM RATIO RESULTS')
    print('='*60)
    print(f'  {"Fraction":<10} {"Mean Win%":>10} {"±Std":>8} {"First Success":>15}')
    print('-'*60)
    for fraction in DREAM_FRACTIONS:
        rlist = [r for r in all_results if r['dream_fraction'] == fraction]
        wpts  = [r['win_pct'] for r in rlist]
        feps  = [r['first_success'] for r in rlist if r['first_success'] is not None]
        avg_f = int(np.mean(feps)) if feps else None
        print(f'  {fraction:<10.2f} {np.mean(wpts):>9.1f}% {np.std(wpts):>7.1f}%  {str(avg_f):>15}')
    print('='*60)

    save_json({'episodes': args.episodes, 'seeds': seeds, 'results': all_results}, 'dream_ratio')
    comp.show(name='dream_ratio_comparison')


if __name__ == '__main__':
    main()
