# Optional app layer (Phases 9–11). Only after the ML model is graded-ready.

## 1. Product name
- **FoodLens** (repo name, keep it). Model codename: **FoodLens-Spoil v1**.
- Alternatives: *PoultryPulse*, *FreshScope*, *SpoilSense*.

## 2. Tech stack
| Layer | Choice | Notes |
|---|---|---|
| Backend | **FastAPI** + Pydantic v2 + Uvicorn, SQLite (SQLModel) | Imports `foodlens` core directly; same result schema as the notebook |
| Web | **Next.js (App Router) + React + TypeScript**, Tailwind CSS v4, **shadcn/ui** (Radix primitives), **assistant-ui** (chat), **Motion** (animations), **Recharts** (charts), TanStack Query | Pin exact versions at Phase 9 start (check latest stable then) |
| Agent protocol | **AG-UI** event stream (tool-call progress → UI), tools exposed as plain functions; optional MCP server wrapping `predict_freshness` | The agent always calls the ML tool. The LLM never "guesses" freshness |
| Android | **Kotlin + Jetpack Compose**, **CameraX + Camera2 interop** for manual ISO/shutter/WB/focus; Ktor client | minSdk **24 (Android 7.0)**; targetSdk = latest required by Google Play (Android 9 / API 28 is too old for Play and lacks newer privacy fixes). Manual exposure needs a camera at Camera2 level FULL/LEVEL_3: detect at runtime and show "manual control not supported" instead of failing |
| iOS (later) | KMP shared logic + SwiftUI camera (AVFoundation manual exposure) | Only if time allows |
| On-device ML (later) | Export features + classical model to ONNX → ONNX Runtime Mobile | The CNN head is optional |

## 3. LLM configuration (no silent failures)
`configs/llm.yaml`:
```yaml
providers:
  ollama_local: {kind: openai_compatible, base_url: http://localhost:11434/v1, api_key_env: null}
  mlx_local:    {kind: openai_compatible, base_url: http://localhost:8080/v1}   # mlx_lm.server / mlx-vlm server
  anthropic:    {kind: anthropic, api_key_env: ANTHROPIC_API_KEY}
  openai:       {kind: openai, api_key_env: OPENAI_API_KEY}
  gemini:       {kind: gemini, api_key_env: GEMINI_API_KEY}
roles:            # each role -> ordered list (first = primary, rest = fallbacks)
  chat:       [{provider: mlx_local, model: gemma-4-e4b-it-4bit, capabilities: [text, image]}]
  reasoning:  [{provider: ollama_local, model: "qwen3.5:9b", capabilities: [text, tools]}]
  vision:     [{provider: mlx_local, model: gemma-4-e4b-it-4bit, capabilities: [image]}]
  stt:        [{provider: local_whisper, model: whisper-large-v3-turbo, capabilities: [audio_in]}]
  tts:        [{provider: local_tts, model: kokoro, capabilities: [audio_out], languages: [en, hi]}]
  embedding:  [{provider: ollama_local, model: "qwen3-embedding:0.6b"}]
retry: {max_attempts: 2, backoff_s: [1, 3], timeout_s: 60}
```
- Before every call, the router checks `capabilities`. If the role needs `image` and the model lacks it, it raises `E_MODEL_CAPABILITY` with the model name. The UI shows a red toast plus a details drawer (provider, model, HTTP status, request id). Errors are never swallowed.
- **Already on this Mac; reuse, do not download again:**
  - Ollama: `qwen3.5:9b`, `nemotron-3.5-bolt:e4b`, `embeddinggemma-2`, `qwen3-embedding:0.6b`, `phonon`.
  - MLX: `~/Library/Application Support/Tavys/models/mlx/mlx-community--gemma-4-e2b-it-4bit` and `…e4b-it-4bit`. Point `mlx_local` at these paths in config; do not copy them into the repo or `.venv`.
- Voice: STT with `mlx-whisper` (Hindi + English; Hinglish works reasonably), TTS with a local Kokoro/MLX-audio voice (verify Hindi voice quality), and the browser Web Speech API as fallback. Cloud live options (Gemini Live, OpenAI Realtime) are selectable in config.

## 4. Key screens
1. **Capture**: live preview with ROI overlay and level/angle indicator, light sequence status (R→G→B), manual ISO/shutter/WB/focus sliders limited to device capabilities, interval timer, and a session recorder that writes the same folder layout as `experiment/`.
2. **Analyse**: drop or capture an R/G/B triplet → progress steps (QC ✓ → ROI ✓ → features ✓ → model ✓) → result card: age ± range, freshness gauge, confidence, "why" (top features), or an **abstain** card with tips.
3. **History**: timeline chart of past results per sample.
4. **Assistant**: chat and live voice with an orb animation and a task list ("Checking image quality…", "Running FoodLens model…").
5. **Settings**: models/providers, temperature source, units, theme.

## 5. Theme (edit `apps/web/app/globals.css` tokens)
| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#F6F8FA` | `#0B1220` | page |
| `--surface-glass` | `rgba(255,255,255,0.65)` + blur 16px | `rgba(255,255,255,0.06)` + blur 16px | cards |
| `--primary` | `#0F9D8A` (teal) | `#2DD4BF` | actions |
| `--fresh` | `#16A34A` | `#22C55E` | fresh state |
| `--warn` | `#D97706` | `#F59E0B` | use soon |
| `--danger` | `#DC2626` | `#F87171` | spoiled risk |
| `--text` | `#0F172A` | `#E2E8F0` | text |
| Fonts | Geist / Inter for UI, JetBrains Mono for numbers | | |
Freshness gauge gradient: `--fresh → --warn → --danger`. Keep contrast ≥ 4.5:1 (WCAG AA) on glass. Respect `prefers-reduced-motion`.

## 6. Global tools: install and uninstall
Anything outside `.venv` must be listed in `scripts/uninstall_global_tools.sh` with its removal command (for example `brew uninstall …`, `ollama rm <model>`, `rm -rf ~/.cache/huggingface/hub/<model>`). The script **prints** the commands and only runs them with `--yes`.
