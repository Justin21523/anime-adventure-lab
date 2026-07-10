# Demo runbook

Target duration: four to six minutes. The committed silent recording is an
80-second overview generated from the same real workbench.

## Seed data

- World ID: `moon-archive`
- World name: 月蝕檔案館
- Player: 凜
- Lore file: a short UTF-8 Markdown document describing the archive gate and
  memory boxes
- Private demo account: `admin`; share the password out-of-band, never in the
  repository or README

## Script

1. Open the public GitHub Pages demo. In 20 seconds explain the problem: an
   author needs story continuity, world-scoped knowledge, and observable AI
   jobs—not another stateless chat box.
2. Open the private workbench and log in. Point out HttpOnly session auth and
   CSRF without dwelling on implementation.
3. Create `moon-archive`, then create 凜's session. This demonstrates the
   World → Session ownership boundary.
4. Submit “調查月光下發亮的檔案盒”. While queued, show the job progress state;
   after completion, show the persisted narrative, choices, state version, and
   turn number.
5. Explain that the request carried an idempotency key and that refresh/retry
   cannot advance the story twice.
6. Expand `RAG 引用證據` and show filename, excerpt, chunk ID, and similarity
   score. Toggle RAG off only to explain the explicit behavior boundary.
7. Open the pending WorldPack proposal, approve it, and show the world version
   increment. AI output never writes authoritative world state directly.
8. Open Job Inspector and Runtime & Services. Point out request ID, execution
   ID, attempt, duration, migration revision, and healthy dependencies.
9. Switch to Engineering Evidence, select the completed job, and walk through
   the persisted API → worker event timeline. Explain that these events share
   the transaction with each state transition and are not reconstructed logs.
10. End with the architecture diagram, generated verification report, and one explicit tradeoff:
   VLM/training/export are experimental profiles so the Story product can make
   an honest production-readiness claim.

## States to capture

- Empty: no worlds, then no sessions, then no turns.
- Loading: API connection, story timeline, and active job progress.
- Success: completed turn with choices and updated session state.
- Error: unsupported lore file and unavailable API.
- Recovery: safe turn retry and durable queued job after dispatch is deferred.
- Engineering evidence: filtered job list, chronological event chain, and the
  explicit unavailable state when an evidence API cannot be reached.

## README media

Run `npm run capture:demo` from `frontend/react` with `DEMO_ADMIN_PASSWORD` set.
The Playwright flow captures dashboard, knowledge, running job, completed turn,
RAG evidence, pending/approved review, system health, mobile layout, and encodes
the actual states into an H.264 video with ffmpeg.
