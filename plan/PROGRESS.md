# PROGRESS: single source of truth for "where are we"

## Current state (rewrite every session, 3–6 lines)
- S01 (2026-11-18, planning): research, first-pass EDA, docs, agent rules, skills and the phase plan written. **No ML code yet.**
- Next: **P0 Foundation** (`plan/phases/P0_foundation.md`).
- Blockers: data-team answers to `docs/04_DATA_COLLECTION_CHANGES.md` §D (meaning of D1/D2/D3, A/B, storage between sessions). P0–P4 can proceed without them; P5 labels need them (defaults are defined).

## Phase checklist
- [x] Planning & research (S01)
- [ ] P0 Foundation: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P1 Ingest & QC: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P2 ROI & glints: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ]
- [ ] P3 EDA: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P4 Features: T1 [ ] T2 [ ] T3 [ ] T4 [ ]
- [ ] P5 Labels & evaluation: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ]
- [ ] P6 Classical models: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ]
- [ ] P7 Deep features (optional-but-recommended): T1 [ ] T2 [ ] T3 [ ] T4 [ ]
- [ ] P8 Inference, notebook, README, report: T1 [ ] T2 [ ] T3 [ ] T4 [ ] T5 [ ] T6 [ ]
- [ ] P9* API · [ ] P10* Web · [ ] P11* Android/voice

## Key numbers (filled by code outputs; never by hand)
| Metric | Value | Source file |
|---|---|---|
| Frames total / usable | 944 / ? | reports/tables/qc_summary.csv |
| Best grouped balanced accuracy (class) | ? | reports/metrics/… |
| Best grouped MAE hours | ? | reports/metrics/… |

## Open questions (for user / data team)
1. Meaning of D1/D2/D3 and A/B; is any piece filmed in two sessions? (blocks `piece_id`, `storage_day`)
2. Storage between EXP-2 sessions (fridge?) and purchase times (blocks absolute age).
3. Freshness class boundaries: default proposal in P5 (confirm with the mentor).

## Recent sessions (newest first; keep 5 in full)
### S01: 2026-11-18: Planning (Claude Opus)
- Did: dataset inventory (944 jpg, 5 sessions), EXIF check (manual ISO100 1/60 f1.7, auto WB), first-pass whole-frame EDA, found the blue-LED failure in D1A (24 black frames) and a scene-change frame, glint trend inconsistency; read Papers 23/24; wrote docs/01–06, AGENTS.md, skills, plan.
- Files: AGENTS.md, CLAUDE.md, GEMINI.md, docs/*, plan/*, .claude/skills/*, reports/eda_initial/*, .gitignore.
- Verified: EDA script ran on all 944 images (24 s). No package code yet, so no check.sh.
- Next: P0.

## Older sessions (one sentence each)
- (none)
