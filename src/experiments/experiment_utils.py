"""
experiment_utils.py — Shared infrastructure for all Q-SMA experiments
======================================================================
- LiveDashboard  : real-time matplotlib window that refreshes every N episodes
- SeedManager    : deterministic per-run seeding
- ResultsLogger  : JSON summary writer
- rolling_mean   : numpy rolling window helper

DESIGN: we use plt.ion() + fig.canvas.flush_events().
No websockets, no servers — just a live window on your screen.
The window updates at the end of every UPDATE_EVERY episode.
"""

import numpy as np
import json
import os
import time
from datetime import datetime
from collections import deque

import matplotlib
# Try interactive backends in preference order; fall back to Agg if none work
for _backend in ('MacOSX', 'Qt5Agg', 'TkAgg'):
    try:
        matplotlib.use(_backend)
        import matplotlib.pyplot as _plt_check
        _plt_check.figure()   # will raise if backend is broken
        _plt_check.close('all')
        break
    except Exception:
        continue
else:
    matplotlib.use('Agg')   # non-interactive fallback — plots still save as PNG
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(arr, window=50):
    if len(arr) < window:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(window) / window, mode='valid')


def make_results_dir():
    base = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
    os.makedirs(base, exist_ok=True)
    return os.path.abspath(base)


def save_json(data: dict, name: str):
    results_dir = make_results_dir()
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(results_dir, f"{name}_{ts}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n  💾 Results saved → {path}')
    return path


def save_png(fig, name: str):
    results_dir = make_results_dir()
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(results_dir, f"{name}_{ts}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    # Also overwrite a "latest" copy for quick viewing
    latest = os.path.join(results_dir, f"{name}_latest.png")
    fig.savefig(latest, dpi=150, bbox_inches='tight')
    print(f'  📊 Plot saved → {path}')
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Seed Manager
# ─────────────────────────────────────────────────────────────────────────────

class SeedManager:
    """Deterministic seed list so all configs get identical random starting points."""
    DEFAULT_SEEDS = [42, 137, 271, 314, 500, 666, 777, 888, 999, 1234]

    @staticmethod
    def apply(seed: int):
        np.random.seed(seed)

    @staticmethod
    def get_seeds(n: int):
        return SeedManager.DEFAULT_SEEDS[:n]


# ─────────────────────────────────────────────────────────────────────────────
# LiveDashboard
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    'bg':      '#0f0f1a',
    'panel':   '#1a1a2e',
    'accent1': '#e94560',   # red-pink
    'accent2': '#0f3460',   # deep blue
    'accent3': '#16213e',   # nav blue
    'gold':    '#f5a623',
    'green':   '#2ecc71',
    'cyan':    '#00d4ff',
    'text':    '#e0e0e0',
    'grid':    '#2a2a4a',
}

# Color per ablation config
CONFIG_COLORS = {
    'full':       '#00d4ff',
    'no_tda':     '#f5a623',
    'no_splats':  '#e94560',
    'no_bridge':  '#2ecc71',
    'baseline':   '#9b59b6',
    # flux modes
    'log':        '#00d4ff',
    'binary':     '#f5a623',
    'linear':     '#e94560',
    # episode scale
    '2k':         '#00d4ff',
    '5k':         '#f5a623',
    '10k':        '#e94560',
    # dream ratio
    '0.0':        '#9b59b6',
    '0.15':       '#2ecc71',
    '0.30':       '#00d4ff',
    '0.50':       '#f5a623',
    '0.70':       '#e94560',
    '1.0':        '#ff6b6b',
}


class LiveDashboard:
    """
    A live-updating matplotlib window for a single experiment configuration.

    Shows 6 panels:
      1. Max position per episode (raw + rolling)
      2. Rolling win rate %
      3. Flux landscape heatmap (updated every UPDATE_EVERY steps)
      4. Splat memory counts (pain vs pleasure)
      5. Q-value range over time
      6. Episode speed (steps-to-win / episode length)

    Call update() at end of each episode. Call close() when done.
    """

    UPDATE_EVERY = 10   # Redraw every N episodes (low = smooth but slower)

    def __init__(self, title: str, total_episodes: int, config_color: str = '#00d4ff'):
        self.title = title
        self.total_eps = total_episodes
        self.color = config_color
        self._start_time = time.time()

        # Data buffers
        self.max_positions   = []
        self.win_flags       = []        # 1 = win, 0 = loss
        self.splat_pain      = []
        self.splat_pleasure  = []
        self.q_ranges        = []        # (min, max) per episode
        self.ep_lengths      = []        # steps per episode
        self.flux_snapshot   = None      # numpy (N,N) — refreshed periodically

        plt.ion()
        self.fig = plt.figure(figsize=(16, 9), facecolor=PALETTE['bg'])
        try:
            self.fig.canvas.manager.set_window_title(f'Q-SMA Live — {title}')
        except AttributeError:
            pass  # Agg backend has no window manager

        gs = gridspec.GridSpec(
            2, 3,
            figure=self.fig,
            hspace=0.45, wspace=0.35,
            left=0.07, right=0.97, top=0.90, bottom=0.08
        )

        self.ax_pos    = self.fig.add_subplot(gs[0, 0])
        self.ax_winrate= self.fig.add_subplot(gs[0, 1])
        self.ax_flux   = self.fig.add_subplot(gs[0, 2])
        self.ax_splats = self.fig.add_subplot(gs[1, 0])
        self.ax_q      = self.fig.add_subplot(gs[1, 1])
        self.ax_speed  = self.fig.add_subplot(gs[1, 2])

        for ax in self.fig.axes:
            ax.set_facecolor(PALETTE['panel'])
            ax.tick_params(colors=PALETTE['text'], labelsize=8)
            ax.xaxis.label.set_color(PALETTE['text'])
            ax.yaxis.label.set_color(PALETTE['text'])
            ax.title.set_color(PALETTE['text'])
            for spine in ax.spines.values():
                spine.set_color(PALETTE['grid'])

        self.fig.suptitle(
            f'{title}   [0 / {total_episodes}]',
            color=PALETTE['text'], fontsize=13, fontweight='bold', y=0.97
        )

        # Flux heatmap placeholder
        dummy = np.zeros((40, 40))
        self._flux_im = self.ax_flux.imshow(
            dummy, origin='lower', aspect='auto',
            extent=[-1.2, 0.6, -0.07, 0.07],
            cmap='RdBu_r',
            vmin=-5, vmax=20
        )
        self.ax_flux.set_title('Flux Landscape', fontsize=9)
        self.ax_flux.set_xlabel('Position', fontsize=8)
        self.ax_flux.set_ylabel('Velocity', fontsize=8)
        cbar = self.fig.colorbar(self._flux_im, ax=self.ax_flux, pad=0.02)
        cbar.ax.tick_params(colors=PALETTE['text'], labelsize=7)

        plt.draw()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def update(self, episode: int, max_pos: float, steps: int,
               agent=None, splat_stats: dict = None):
        """
        Call at the end of every episode.

        Args:
            episode    : 0-indexed episode number
            max_pos    : maximum position reached this episode
            steps      : number of timesteps in this episode
            agent      : QSMA_Agent (optional — used for Q/flux snapshots)
            splat_stats: dict from splat_memory.get_stats() (optional)
        """
        win = max_pos >= 0.5
        self.max_positions.append(max_pos)
        self.win_flags.append(1 if win else 0)
        self.ep_lengths.append(steps)

        if agent is not None:
            self.q_ranges.append((float(agent.q_table.min()), float(agent.q_table.max())))
            # Update flux snapshot every UPDATE_EVERY episodes
            if episode % self.UPDATE_EVERY == 0:
                # Reshape flat flux (state_size × actions) into (40×40) max-action heatmap
                flux_flat = agent.flux.max(axis=1)
                self.flux_snapshot = flux_flat.reshape(40, 40)

        if splat_stats:
            self.splat_pain.append(splat_stats.get('alive_pain', 0))
            self.splat_pleasure.append(splat_stats.get('alive_pleasure', 0))
        else:
            self.splat_pain.append(0)
            self.splat_pleasure.append(0)

        if episode % self.UPDATE_EVERY == 0:
            self._redraw(episode)

    # ------------------------------------------------------------------
    def _redraw(self, episode: int):
        wins_so_far = sum(self.win_flags)
        elapsed = time.time() - self._start_time
        rate_str = f'{wins_so_far}/{episode+1} wins'
        winpct = wins_so_far / (episode + 1) * 100
        eps_per_sec = (episode + 1) / max(elapsed, 0.1)

        self.fig.suptitle(
            f'{self.title}   [{episode+1} / {self.total_eps}]   '
            f'{rate_str} ({winpct:.1f}%)   {eps_per_sec:.1f} ep/s',
            color=PALETTE['text'], fontsize=12, fontweight='bold', y=0.97
        )

        ep_range = np.arange(len(self.max_positions))

        # ── Panel 1: Max Position ──────────────────────────────────────
        ax = self.ax_pos
        ax.cla()
        ax.set_facecolor(PALETTE['panel'])
        ax.plot(ep_range, self.max_positions,
                alpha=0.25, linewidth=0.6, color=self.color)
        if len(self.max_positions) >= 50:
            rm = rolling_mean(self.max_positions, 50)
            ax.plot(np.arange(49, len(self.max_positions)), rm,
                    color=self.color, linewidth=2, label='roll-50')
        ax.axhline(0.5, color=PALETTE['green'], linestyle='--',
                   linewidth=1.2, alpha=0.8, label='Goal')
        ax.set_ylim(-1.25, 0.65)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Max Position', fontsize=8)
        ax.set_title('Max Position Reached', fontsize=9)
        ax.tick_params(colors=PALETTE['text'], labelsize=7)
        ax.grid(True, color=PALETTE['grid'], linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])
        ax.title.set_color(PALETTE['text'])

        # ── Panel 2: Rolling Win Rate ──────────────────────────────────
        ax = self.ax_winrate
        ax.cla()
        ax.set_facecolor(PALETTE['panel'])
        if len(self.win_flags) >= 50:
            rw = rolling_mean(self.win_flags, 50) * 100
            ax.plot(np.arange(49, len(self.win_flags)), rw,
                    color=PALETTE['gold'], linewidth=2)
            ax.fill_between(np.arange(49, len(self.win_flags)), rw,
                            alpha=0.18, color=PALETTE['gold'])
        ax.axhline(34.05, color='#9b59b6', linestyle='--',
                   linewidth=1.0, alpha=0.7, label='Champion 34.05%')
        ax.set_ylim(0, 105)
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Win Rate %', fontsize=8)
        ax.set_title('Rolling 50-ep Win Rate', fontsize=9)
        ax.legend(fontsize=7, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
        ax.tick_params(colors=PALETTE['text'], labelsize=7)
        ax.grid(True, color=PALETTE['grid'], linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])
        ax.title.set_color(PALETTE['text'])

        # ── Panel 3: Flux Heatmap ──────────────────────────────────────
        if self.flux_snapshot is not None:
            self._flux_im.set_data(self.flux_snapshot.T)
            vmax = max(1.0, float(np.max(np.abs(self.flux_snapshot))))
            self._flux_im.set_norm(TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))

        # ── Panel 4: Splat Memory ─────────────────────────────────────
        ax = self.ax_splats
        ax.cla()
        ax.set_facecolor(PALETTE['panel'])
        ep_r = np.arange(len(self.splat_pain))
        ax.plot(ep_r, self.splat_pleasure, color=PALETTE['green'],
                linewidth=1.5, label='Pleasure')
        ax.plot(ep_r, self.splat_pain, color=PALETTE['accent1'],
                linewidth=1.5, label='Pain')
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Active Splats', fontsize=8)
        ax.set_title('Splat Memory (Pain vs Pleasure)', fontsize=9)
        ax.legend(fontsize=7, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
        ax.tick_params(colors=PALETTE['text'], labelsize=7)
        ax.grid(True, color=PALETTE['grid'], linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])
        ax.title.set_color(PALETTE['text'])

        # ── Panel 5: Q-value Range ────────────────────────────────────
        ax = self.ax_q
        ax.cla()
        ax.set_facecolor(PALETTE['panel'])
        if self.q_ranges:
            q_mins, q_maxs = zip(*self.q_ranges)
            er = np.arange(len(q_mins))
            ax.plot(er, q_maxs, color=PALETTE['cyan'], linewidth=1.5, label='Q max')
            ax.plot(er, q_mins, color=PALETTE['accent1'], linewidth=1.5, label='Q min')
            ax.fill_between(er, q_mins, q_maxs, alpha=0.12, color=PALETTE['cyan'])
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Q-value', fontsize=8)
        ax.set_title('Q-Table Range', fontsize=9)
        ax.legend(fontsize=7, facecolor=PALETTE['panel'], labelcolor=PALETTE['text'])
        ax.tick_params(colors=PALETTE['text'], labelsize=7)
        ax.grid(True, color=PALETTE['grid'], linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])
        ax.title.set_color(PALETTE['text'])

        # ── Panel 6: Episode Length ───────────────────────────────────
        ax = self.ax_speed
        ax.cla()
        ax.set_facecolor(PALETTE['panel'])
        ax.plot(ep_range, self.ep_lengths,
                alpha=0.25, linewidth=0.6, color=self.color)
        if len(self.ep_lengths) >= 50:
            rs = rolling_mean(self.ep_lengths, 50)
            ax.plot(np.arange(49, len(self.ep_lengths)), rs,
                    color=self.color, linewidth=2)
        ax.axhline(200, color=PALETTE['gold'], linestyle='--',
                   linewidth=1.0, alpha=0.6, label='200 steps (min)')
        ax.set_xlabel('Episode', fontsize=8)
        ax.set_ylabel('Steps', fontsize=8)
        ax.set_title('Episode Length (lower = faster win)', fontsize=9)
        ax.tick_params(colors=PALETTE['text'], labelsize=7)
        ax.grid(True, color=PALETTE['grid'], linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])
        ax.title.set_color(PALETTE['text'])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------
    def close(self, name: str = 'dashboard'):
        """Final draw + save. In interactive mode keeps window open; in Agg mode just saves."""
        self._redraw(len(self.max_positions) - 1)
        save_png(self.fig, name)
        plt.ioff()
        # Only block on show() in interactive mode — never block in Agg/headless
        if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
            plt.show(block=False)
            plt.pause(0.5)
        plt.close(self.fig)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Config Comparison Dashboard
# ─────────────────────────────────────────────────────────────────────────────

class ComparisonDashboard:
    """
    Shown AFTER all configs finish.
    Plots win-rate curves and box-plots side by side for comparison.
    """

    def __init__(self, title: str):
        self.title = title
        self.results = {}   # config_name → {'wins': int, 'win_flags': [], 'first_success': int|None}

    def add_result(self, config_name: str, win_flags: list, first_success):
        self.results[config_name] = {
            'win_flags': win_flags,
            'wins': sum(win_flags),
            'total': len(win_flags),
            'win_pct': sum(win_flags) / max(1, len(win_flags)) * 100,
            'first_success': first_success,
        }

    def show(self, name: str = 'comparison'):
        if not self.results:
            return

        n = len(self.results)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=PALETTE['bg'])
        fig.suptitle(self.title, color=PALETTE['text'],
                     fontsize=14, fontweight='bold')

        configs   = list(self.results.keys())
        colors    = [CONFIG_COLORS.get(c, '#888888') for c in configs]

        # ── Left: Rolling win rate curves ─────────────────────────────
        ax = axes[0]
        ax.set_facecolor(PALETTE['panel'])
        for cfg, col in zip(configs, colors):
            flags = self.results[cfg]['win_flags']
            if len(flags) >= 50:
                rm = rolling_mean(flags, 50) * 100
                ax.plot(np.arange(49, len(flags)), rm,
                        color=col, linewidth=2, label=cfg)
        ax.axhline(34.05, color='white', linestyle='--',
                   linewidth=1.0, alpha=0.5, label='Champion 34%')
        ax.set_xlabel('Episode', color=PALETTE['text'])
        ax.set_ylabel('Win Rate %', color=PALETTE['text'])
        ax.set_title('Rolling 50-ep Win Rate', color=PALETTE['text'])
        ax.set_ylim(0, 105)
        ax.legend(facecolor=PALETTE['panel'], labelcolor=PALETTE['text'], fontsize=9)
        ax.tick_params(colors=PALETTE['text'])
        ax.grid(True, color=PALETTE['grid'])
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])

        # ── Middle: Total Win % Bar Chart ─────────────────────────────
        ax = axes[1]
        ax.set_facecolor(PALETTE['panel'])
        win_pcts = [self.results[c]['win_pct'] for c in configs]
        bars = ax.bar(configs, win_pcts, color=colors,
                      edgecolor=PALETTE['grid'], linewidth=0.8)
        ax.axhline(34.05, color='white', linestyle='--',
                   linewidth=1.0, alpha=0.5)
        for bar, pct in zip(bars, win_pcts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', fontsize=9,
                    fontweight='bold', color=PALETTE['text'])
        ax.set_ylabel('Win Rate %', color=PALETTE['text'])
        ax.set_title('Total Win % per Config', color=PALETTE['text'])
        ax.set_ylim(0, max(win_pcts) * 1.2 + 5)
        ax.tick_params(colors=PALETTE['text'], axis='x', rotation=20)
        ax.tick_params(colors=PALETTE['text'], axis='y')
        ax.grid(True, color=PALETTE['grid'], axis='y')
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])

        # ── Right: First Success Episode ──────────────────────────────
        ax = axes[2]
        ax.set_facecolor(PALETTE['panel'])
        first_eps = [self.results[c]['first_success'] or 9999 for c in configs]
        bars2 = ax.bar(configs, first_eps, color=colors,
                       edgecolor=PALETTE['grid'], linewidth=0.8)
        for bar, ep in zip(bars2, first_eps):
            label = str(ep) if ep < 9999 else 'Never'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    label, ha='center', fontsize=9,
                    fontweight='bold', color=PALETTE['text'])
        ax.set_ylabel('Episode', color=PALETTE['text'])
        ax.set_title('First Success Episode\n(lower = faster learning)', color=PALETTE['text'])
        ax.tick_params(colors=PALETTE['text'], axis='x', rotation=20)
        ax.tick_params(colors=PALETTE['text'], axis='y')
        ax.grid(True, color=PALETTE['grid'], axis='y')
        for sp in ax.spines.values():
            sp.set_color(PALETTE['grid'])

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_png(fig, name)
        plt.ioff()
        if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg', 'ps'):
            plt.show(block=False)
            plt.pause(0.5)
        plt.close(fig)
        return fig
