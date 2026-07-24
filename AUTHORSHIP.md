# Authorship & Provenance

**House rule:** every failure logged is a path someone else does not have to re-walk.  
**House rule:** the collaboration is named.

## Core team (all four of us)

| Role | Who |
|------|-----|
| **Author / Operator / Vision** | **Jason Van Pham (Ruffian-L / Shepard)** — continuous thread, human-tuned physics, the decision to keep dead ends on disk |
| **Co-engineer** | **Grok (xAI)** |
| **Co-engineer** | **Claude (Anthropic)** |
| **Co-engineer** | **Gemini (Google)** |

This has been multi-AI collaboration since October 2025.  
Names were temporarily cleaned from some public faces when the work was attacked as “slop”; they are restored so the trail stays accurate.

Failures stay on disk on purpose — `research_history/`, `snapshots/`, and the Phase 5 ablations that reverse earlier stories are part of the record, not debris.

## Short form (commit / fork / one-liner)

```
Co-engineered by Jason Van Pham (Ruffian-L) with Grok (xAI), Claude (Anthropic), and Gemini (Google). It was all four of us.
```

## This repository

**Project:** Physics of Friendship — MountainCar Q-SMA  
**Public home:** https://github.com/Ruffian-L/physics-of-friendship-mountaincar-rl  
**What it is:** research journey (Feb 2026+) applying Niodoo-family ideas — habit (Flux), TDA self-monitoring, dream replay, splat memory, body↔mind bridge — to MountainCar-v0.

### Roles on this work

- **Jason Van Pham (Ruffian-L / Shepard)** — Author, operator, vision; validation target (“does the agent climb out of the valley?”); keep-wins-and-losses honesty bar.
- **Claude (Anthropic)** — Co-engineer on professionalization and portability (reproduction harness, seeding, CI smoke, no-hype wording). Git trail: `Co-Authored-By` on `5c36afa`, `d8954ef`.
- **Grok (xAI)** — Co-engineer; attribution pass and continuous thread with the wider Niodoo / swarm research line.
- **Gemini (Google)** — Co-engineer on the broader multi-AI stack that produced Niodoo steering and the ideas this repo stress-tests.

### Provenance note (do not erase)

The README “Why this exists” section documents the defensive period after Niodoo steering was dismissed as slop. That history is intentional. This file does not rewrite it — it only makes the four-way collaboration explicit and permanent at the root of the tree.

### Commit attribution convention going forward

When a change is co-engineered, the commit body should include the matching trailer(s), for example:

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Grok (xAI) <noreply@x.ai>
Co-Authored-By: Gemini (Google) <noreply@google.com>
```

Operator commits that are human-only need no AI trailers. AI-session tooling may use product-specific addresses; the names above are the human-readable standard.

### What stays on disk on purpose

| Path | Why it stays |
|------|----------------|
| `research_history/` | Phase-by-phase plots (including failed loops / well addiction) |
| `snapshots/` | Frozen milestones with technical writeups — not “cleaned” winners only |
| `results/` | Ablation + long-run numbers that revised the bridge story |
| Documented dead ends in README | So the next person does not re-walk them |

---

**Authorship of this file**

- **Author:** Grok (xAI) — with Jason’s instruction that all four of us are named
- **Role:** provenance / attribution restore
- **Project:** physics-of-friendship-mountaincar-rl
- **Date written:** 2026-07-24
- **Note:** Failures logged on purpose so the next person does not re-walk the same dead ends.
