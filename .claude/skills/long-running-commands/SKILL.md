---
name: long-running-commands
description: How to run slow scripts (check.sh, feature extraction, training) without skipping, giving up early, or waiting blindly on an error.
---
# Long-running commands

1. Always smoke-test first: `--limit 20`.
2. Full run with a log: `mkdir -p logs && <cmd> 2>&1 | tee logs/<stage>_$(date +%H%M).log`. In tools that block for at most N minutes, run in the background (`nohup <cmd> > logs/x.log 2>&1 &`) and poll.
3. Poll every 30-60 s: `tail -n 25 logs/x.log`. Expected durations: check.sh 2-10 min, ROI over all images 5-15 min, features 5-20 min, CNN embeddings 10-30 min on MPS per backbone, CNN fine-tuning (nested, 9 outer folds x 3 seeds) several hours: estimate from the `--limit 20` smoke run, run it in the background, checkpoint per outer fold so a killed run resumes instead of restarting.
4. On each poll scan for `Traceback|Error|FAILED|Killed|nan|MemoryError`. If found: kill the job (`pkill -f <script>`), read about 40 lines around the error, fix the root cause, add a regression test, rerun from the start of that stage.
5. Done = the process has exited, the exit code is 0, AND the final marker line is present (`CHECK RESULT: PASS` / `STAGE DONE <stage> rows=<n>`). Otherwise it is not done.
6. Never report "should pass" or "likely works". Report the actual tail of the log.
7. If a run is too slow, reduce the work properly (resolution from config, `n_jobs`, caching). Do not silently drop data.
