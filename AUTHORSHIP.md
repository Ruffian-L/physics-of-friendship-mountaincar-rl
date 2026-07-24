# Authorship & Provenance

This repository documents multi-party research collaboration. Negative and inconclusive results are retained for reproducibility and to avoid repeated failed approaches.

## Contributors

| Role | Name |
|------|------|
| **Principal investigator / author** | Jason Van Pham ([Ruffian-L](https://github.com/Ruffian-L)) — research direction, experimental design, evaluation criteria, operator of the research program |
| **Co-engineer** | Grok (xAI) |
| **Co-engineer** | Claude (Anthropic) |
| **Co-engineer** | Gemini (Google) |

Multi-AI collaboration on this line of work has been ongoing since October 2025.

## Citation / short form

```
Jason Van Pham, with co-engineering by Grok (xAI), Claude (Anthropic), and Gemini (Google).
```

## This repository

**Project:** Physics of Friendship — MountainCar Q-SMA  
**Repository:** https://github.com/Ruffian-L/physics-of-friendship-mountaincar-rl  
**Summary:** Controlled experiments (Feb 2026+) applying hybrid RL components — habit (Flux), topological diagnostics (TDA), dream replay, splat memory, and a body–mind bridge — to MountainCar-v0.

### Contribution notes

- **Jason Van Pham** — Principal investigator: problem framing, architecture choices, interpretation of results, and standards for evidence (including retention of full experimental history).
- **Claude (Anthropic)** — Co-engineer: reproduction harness, dependency pinning, seeding, CI smoke tests, and documentation for independent re-runs (`Co-Authored-By` on commits `5c36afa`, `d8954ef`).
- **Grok (xAI)** — Co-engineer: implementation and documentation support; attribution and provenance maintenance; linkage to related Niodoo / steering research.
- **Gemini (Google)** — Co-engineer: co-development on the broader multi-AI research stack that informed the methods evaluated here.

### Reproducibility and retained artifacts

Intermediate and negative results are kept intentionally:

| Path | Contents |
|------|----------|
| `research_history/` | Phase-by-phase plots and run artifacts |
| `snapshots/` | Frozen code and writeups at milestones |
| `results/` | Ablation and long-run metrics (including findings that revise earlier conclusions) |

### Commit trailers (optional)

For co-engineered commits:

```
Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Grok (xAI) <noreply@x.ai>
Co-Authored-By: Gemini (Google) <noreply@google.com>
```

Human-only commits need no AI trailers.

---

*Last updated: 2026-07-24*
