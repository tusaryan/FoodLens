# HANDOFF: next-session prompt template

At the end of each session, the agent fills in the `{{…}}` placeholders and prints the result as its last message. Keep it **under ~40 lines**: point to files, do not paste their contents.

```text
You are continuing the FoodLens project (ML model that estimates room-temperature age / freshness of raw chicken from red/green/blue-LED smartphone photos; the leakage-free ML model + honest evaluation is the graded deliverable, apps are optional).

Working directory: /Users/aryan/FoodLens

Before anything else, read IN THIS ORDER and nothing else unless needed:
1. AGENTS.md (mandatory rules: session start/end protocol, no leakage, file-size limits, wait for long commands)
2. plan/PROGRESS.md (current state, open questions)
3. plan/LESSONS.md (pitfalls)
4. plan/phases/{{PHASE_FILE}}  ← this session's phase
5. Skills listed in that phase file under "Skills to load" (.claude/skills/<name>/SKILL.md)

Last session: {{SESSION_ID}} on {{DATE}}: {{ONE_LINE_SUMMARY}}
Last commit: {{COMMIT_HASH}} "{{COMMIT_MSG}}"; check.sh status: {{PASS/FAIL + note}}

This session's goal: complete {{TASK_IDS}} of {{PHASE}}.
Carry-over / unfinished: {{CARRY_OVER or "none"}}
Watch out for: {{1–3 specific pitfalls from LESSONS relevant to these tasks}}
Open questions affecting this session: {{or "none"}}

Attachments (only if your tool supports @-mentions; otherwise read the paths above):
@AGENTS.md @plan/PROGRESS.md @plan/LESSONS.md @plan/phases/{{PHASE_FILE}}

Follow the session protocol in AGENTS.md: run scripts/check.sh first, state a 5-line plan, implement every listed sub-step (no shortcuts), wait for long commands and fix errors you see in their logs, then finish with the end-of-session checklist (check.sh PASS, PROGRESS/LESSONS/session log, git commit, next handoff prompt).
```

## Tips for the user
- Use a strong model (Claude Opus/Sonnet, GPT-5-class) for P2, P5, P6 (the science-heavy phases). Faster models are fine for P0, P4 and the docs work in P8.
- If a session ends abruptly (credits/context), start the next one with the same prompt plus: "Previous session ended abruptly; run `git status`/`git diff --stat`, reconcile with PROGRESS.md, then continue."
