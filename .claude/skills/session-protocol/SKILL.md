---
name: session-protocol
description: Start and end every FoodLens work session correctly - what to read, verification, PROGRESS/LESSONS/session-log updates, commit, and next-session prompt generation.
---
# Session protocol (FoodLens)

## Start
1. Read `plan/PROGRESS.md`, `plan/LESSONS.md`, then ONLY the phase file named in the prompt.
2. `git status` + `git log --oneline -5`. Unknown uncommitted changes: stop and ask.
3. `bash scripts/check.sh 2>&1 | tee logs/check_latest.log`. Must end with `CHECK RESULT: PASS`. If not, fix it first.
4. State your plan in at most 5 lines and name the task IDs (e.g. `P2.T3`) you will complete this session. Pick at most what fits about 60 % of your context; leave the rest for the next session.

## During
- Work task by task. After each task: run its acceptance check (listed in the phase file), tick it in PROGRESS.md immediately (so a crash loses nothing).
- Keep reads targeted: `grep -n`, `sed -n 'a,bp'`. Do not cat whole large files.

## End (all mandatory)
1. `bash scripts/check.sh` → PASS (paste last 5 lines).
2. Smoke-run changed stages with `--limit 20`.
3. PROGRESS.md: tick tasks, rewrite "Current state" (3-6 lines), add the session entry at the top of "Recent sessions". If more than 5 entries exist, compress the oldest into ONE sentence under "Older sessions".
4. LESSONS.md: append dated one-liners for new pitfalls/insights.
5. `plan/sessions/S<NN>_<date>_<phase>.md`: full log (template in `plan/sessions/_TEMPLATE.md`).
6. `git add -A && git commit -m "feat(P<n>): <summary>"`. No Claude/AI/agent mentions or Co-Authored-By trailers in commit messages.
7. Fill `plan/HANDOFF.md` template and print the prompt as the final block of your reply.

## Never
- Never mark a task done that you did not run and verify.
- Never "simplify" a task by skipping listed sub-steps. If a sub-step is impossible, record why in Open questions.
