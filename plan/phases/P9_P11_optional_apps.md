# P9–P11: Optional app layer (only after P8 is done and the user says go)

Design reference: `docs/06_APP_DESIGN_OPTIONAL.md`. Architecture: `docs/03_ARCHITECTURE.md` §4–5. Before starting, check the current stable versions of each framework (web search) and pin them exactly.

## P9: FastAPI backend (`apps/api/`, 1 session)
- T1 `POST /v1/predict` (multipart 1–3 images + optional light tags/temp) → `PredictionResult` JSON (the same schema as P8). `GET /v1/health` (model version, config hash). `GET /v1/model-card`.
- T2 `POST /v1/captures` and `GET /v1/captures/{session}` storing capture sessions in SQLite + the images on disk using the `experiment/`-style layout, so new data flows straight into P1 ingest.
- T3 Errors: a FoodLensError → HTTP 422 with `{code, message, hint, request_id}`; unexpected → 500 with request_id, logged. Never a 200 with an empty result.
- T4 Tests with `httpx`/TestClient; latency budget < 2 s per triplet on M1.
- T5 `apps/api/README.md`, run with `uv run uvicorn …`.

## P10: Web app (`apps/web/`, 2–3 sessions)
- Next.js + TS + Tailwind + shadcn/ui + Motion + Recharts + assistant-ui; tokens from docs/06 §5; light/dark toggle.
- Screens: Analyse (upload/capture triplet → animated step progress → result card/gauge/abstain card), History, Capture (browser `getUserMedia`; note: browsers cannot set ISO/shutter; show a warning and recommend the Android app for data collection), Assistant (chat + voice), Settings (LLM roles from `configs/llm.yaml` via API).
- Agent: a server-side tool-calling loop (FastAPI endpoint `/v1/agent` streaming AG-UI events): tools `predict_freshness`, `explain_result`, `capture_tips`. The LLM router enforces capabilities, retry/backoff and fallback; errors are streamed as typed events and shown in the UI.
- Tests: Playwright smoke test (upload sample → result card visible), plus lint/typecheck in check.sh (web section only if `apps/web` exists).

## P11: Android capture app + voice (3+ sessions)
- Kotlin + Compose, minSdk 24, targetSdk = current Play requirement. CameraX with Camera2 interop: query `REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR`; if supported, expose ISO/exposure/WB/focus locks; else show "manual control not supported on this device" (never silently use auto).
- Interval capture with a configurable light schedule (R/G/B order, period), on-screen light cue, EXIF preserved, upload to `/v1/captures` or save locally for later sync. Foreground service + wake lock for multi-hour sessions; battery/thermal warnings.
- Voice: push-to-talk and live mode; STT/TTS via the backend roles (Hindi/English/Hinglish); live orb animation; a task list showing the agent steps.
- iOS/KMP only if time remains.
