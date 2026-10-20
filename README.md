# VIGIL X

### Conversational Multi-Camera Video Intelligence

**See Everything. Ask Anything.** · *Ask. Search. Verify. See.*

HackNex 2026 · Problem statement **HNX26EPS05 — Multi-Stream Video Intelligence with Conversational Query**

---

## What is VIGIL X?

VIGIL X is an AI investigator for **recorded** camera footage. You record videos (phones work fine), register each
one against a camera (CAM-01, CAM-02, …) with its real recording start time, and VIGIL X processes the actual video:

* detects objects with **Ultralytics YOLO11m**,
* follows them with **anonymous ByteTrack / BoT-SORT tracks** (`PERSON-TRACK-001`, `CAR-TRACK-004`, …),
* classifies **day / low-light / night** lighting and enhances *analysis copies* of dark frames,
* generates **deterministic events** (person entered/exited view, vehicle detected, person with backpack, …),
* extracts **actual evidence frames** and **short clips** from the original video,
* indexes frames and object crops with **SigLIP 2** image–text embeddings in **FAISS**,
* optionally recognises **consenting, enrolled adult demo participants** by face (everyone else stays **UNKNOWN PERSON**),

and then lets you **ask questions in plain language** — "*Did a red car pass through the main gate?*",
"*What about the staircase?*", "*Did the same person later appear at Lab 1?*", "*Show Demo Person A's journey*",
"*What changed at CAM-01 between 9:00 and 9:30?*" — with every answer backed by a real evidence frame, an
**Evidence Confidence** score, a **coverage** report, **counter-evidence** and a **ProofGraph** you can inspect.

### The no-hallucination rule

The database and the video evidence are the source of truth — **never** the language model.

```
USER QUESTION → QUERY PARSER → STRUCTURED QUERY → DATABASE / FAISS SEARCH → DETECTIONS / TRACKS / EVENTS / FACES
   → ACTUAL EVIDENCE FRAME / CLIP → (top-K VLM verification, cached) → PROOFGRAPH + COVERAGE + COUNTER-EVIDENCE
   → GROUNDED ANSWER
```

* If nothing reliable is found, VIGIL X answers **"No verified evidence was found in the uploaded footage."**
* If the footage is too poor for the requested detail: **"Insufficient visual evidence to reliably identify the requested detail."**
* Every answer gets an evidence status: **CONFIRMED / LIKELY / UNCERTAIN / INSUFFICIENT EVIDENCE**.
* LLM summaries are only used for verified answers and are rejected by a **grounding check** if they mention any
  time, camera, track id, percentage **or count** that is not in the deterministic answer / retrieved evidence.
* VLM output is schema-validated and normalised (`yes | no | uncertain` + confidence + observation); malformed or
  speculative replies (identity, age, intent, "suspicious", …) are marked invalid. A VLM verdict never becomes a fact.
* Semantic (SigLIP) matches not confirmed by the detector, colour analysis or a VLM are labelled **POSSIBLE · UNVERIFIED**.
* Anonymous cross-camera continuity is **VERIFIED / LIKELY / UNCERTAIN / NOT VERIFIED** — appearance consistency, never identity.

---

## Features

| Area | What you get |
|---|---|
| Ingestion | Website upload (drag & drop, real byte progress) **or** drop files in `data/videos/` → **SCAN VIDEOS** → confirm camera/date/start time/timezone → **PROCESS ALL** |
| Formats | MP4 · MOV · AVI · MKV (FFmpeg/OpenCV). Non-H.264 files get a derived browser playback copy; originals are never modified |
| Duplicates | SHA-256 of every file; re-imports are refused. Reprocessing only on explicit request |
| Timestamps | Every detection/event/evidence has a **video-relative** time (`00:02:31`) and an **absolute** time (`2026-10-08 10:02:31 +05:30`) |
| Processing | Background worker; job states **QUEUED / PROCESSING / VERIFYING / INDEXING / COMPLETED / FAILED / RETRYING**; restart-safe; CUDA with CPU fallback |
| Detection / tracking | YOLO11m (configurable) + ByteTrack/BoT-SORT; anonymous IDs; entry/exit edges; motion direction |
| Lighting | DAY / LOW_LIGHT / NIGHT per frame; CLAHE + gamma + denoise on analysis copies only |
| Events | PERSON_DETECTED/ENTERED/EXITED, VEHICLE/BICYCLE_DETECTED, PERSON_WITH_BACKPACK/BAG, MULTIPLE_PEOPLE, OBJECT_MOVEMENT, CROSS_CAMERA_CONTINUITY |
| Evidence | Raw frame, annotated frame, enhanced copy, crop, H.264 clip; viewer with **face boxes, previous/next indexed frame, on-demand VLM verification, download** |
| Search | SQL filters + detector evidence + SigLIP/FAISS → **local VLM verification of the top 6 candidates (Qwen3-VL via Ollama), cached per frame + claim** |
| ProofGraph | Per answer: status, **Evidence Confidence** (weighted, explained), **coverage** (cameras / requested time / frames, GREEN/YELLOW/RED), **“Why did this match?”**, **Challenge answer** (counter-evidence), query-understanding panel, measured performance panel, evidence graph |
| Identity (consent-based) | Enrollment page (adult + consent confirmation, photo preview with face box, embeddings only, delete / re-enroll / pause); per-track identity smoothing; **UNKNOWN PERSON** for everyone else |
| Identity questions | *Where was X?* · *Which cameras saw X?* · *When did X enter CAM-03?* · *How long was X visible?* · *What happened before X entered the lab?* · *Show X's journey* · *Did X move from the entrance to Lab 1?* · *Show unknown people near the staircase* |
| Topology & journeys | `camera_connections` (min/max travel seconds, one/two-way); journeys **CONFIRMED / LIKELY / UNCERTAIN**; topology violations and simultaneous sightings are reported as counter-evidence |
| Watches & alerts | Standing queries replayed on **recorded** footage after each video is processed; alerts OPEN → ACKNOWLEDGED → RESOLVED, de-duplicated, linked to evidence, labelled RECORDED |
| Investigations | Case workspace: attach results/queries/evidence, USER NOTE / AI-GENERATED SUMMARY / SYSTEM OBSERVATION; report as JSON, CSV or printable HTML (save as PDF) separating facts from interpretations |
| Analytics | Track counts by camera / class / hour, peak people per frame, **activity baselines (NORMAL / ELEVATED / UNUSUAL — volume only, never wrongdoing)**, **what changed** between two moments |
| Timeline | All cameras on one time axis: footage coverage, events, enrolled-participant sightings, alerts |
| Feedback | Correct / incorrect / not sure per result — recorded for precision statistics, **no silent retraining** |
| Operations | Model management (AVAILABLE / LOADING / READY / NOT CONFIGURED / ERROR), GPU + latency metrics, camera health (RECORDED SOURCE), privacy controls, **Jury mode** (full-screen presentation) |
| UI pages | Dashboard · Search · Cameras · Videos · Events · Timeline · Identities · Journeys · ProofGraph · Investigations · Alerts · Analytics · Conversations · Data Explorer · Models · Settings · Jury mode |
| Export | CSV / JSON of any query's exact results (with evidence status and coverage), or of the event table |

---

## Architecture

```
                                   VIGIL X
                                      │
                              ┌───────▼───────┐
                              │ User Question │
                              └───────┬───────┘
                              ┌───────▼───────┐  deterministic parser (+ LLM fallback, validated);
                              │ Query Parser  │  enrolled names matched exactly, never inferred
                              └───────┬───────┘
          ┌────────────────┬──────────┼───────────┬──────────────────┐
     Time filter     Camera memory  Semantic   Identity (enrolled)  Topology
          └────────────────┴──────────┼───────────┴──────────────────┘
                              ┌───────▼───────┐  SQLite: tracks / events / detections / faces
                              │ FAISS + SigLIP│  FAISS: frame + object-crop vectors
                              └───────┬───────┘
                               top-K candidates
                              ┌───────▼───────┐  Ollama qwen3-vl:4b (local, cached verdicts) — or
                              │  VLM verify   │  OpenAI-compatible / Gemini / Claude / transformers
                              └───────┬───────┘
                              ┌───────▼───────┐  Evidence Confidence · coverage · why-matched ·
                              │  ProofGraph   │  counter-evidence · evidence graph
                              └───────┬───────┘
                              ┌───────▼───────┐  deterministic answer (+ grounding-checked LLM rephrase)
                              │Grounded answer│
                              └───────┬───────┘
                              ┌───────▼───────┐
                              │   VIGIL X UI  │
                              └───────────────┘
```

**Processing pipeline (per video):** probe → playback copy (if needed) → sampled frames @ `TRACKING_FPS` →
lighting → (enhancement) → YOLO (batched) → ByteTrack → index frames @ `TARGET_FPS` → detections →
**faces on large-enough person boxes (YuNet + SFace, CPU) → gallery decision** → SigLIP embeddings → track
aggregation → **identity smoothing per track (VERIFYING)** → events → evidence frames + clips → FAISS update
(INDEXING) → cross-camera candidates → **watches replayed on the new footage**.

---

## Project structure

```
VIGIL-X/  (this folder)
├── backend/
│   ├── main.py              FastAPI app (serves the built UI; starts/stops the portable Ollama)
│   ├── config.py            settings from .env
│   ├── database.py          SQLite engine (WAL), init + default cameras
│   ├── migrations.py        additive migrations (backup → new tables/columns → schema_migrations log)
│   ├── models/              ORM: cameras, videos, frames, detections, tracks, events, evidence, conversations,
│   │                        identity (identities, references, face observations, track links),
│   │                        investigation (topology, watches, alerts, feedback, investigations, verifications)
│   ├── schemas/             Pydantic request schemas + serializers
│   ├── routes/              cameras(+topology, health), videos, events, evidence, search, chat, dashboard,
│   │                        explorer, system(+models, metrics, privacy), identities, investigations, watches, analytics
│   ├── services/            detector, tracking, lighting, embedding, vector_search, query_parser, search, event,
│   │                        evidence, llm, vlm, ollama, verification, face_recognition, identity, topology,
│   │                        investigation, proof, analytics, watch, report, memory, cross_camera, video, device
│   ├── utils/               time, files (safe paths, hashing), colours, media (FFmpeg)
│   ├── workers/             processing queue + pipeline
│   └── requirements.txt
├── frontend/                React + Vite + Tailwind UI
├── data/
│   ├── videos/              ORIGINAL footage (never modified)
│   ├── backups/             automatic pre-migration database backups
│   ├── processed/ frames/ evidence/ clips/ thumbnails/
│   └── vigil_x.db           SQLite database
├── database/                db_tools.py (stats, integrity, backup, schema)
├── desktop/                 native-window launcher (pywebview / WebView2)
├── models/                  yolo11m.pt · siglip2-base-patch16-224/ · face/ (YuNet, SFace) · ollama/ (qwen3-vl:4b)
├── tools/ollama/            portable Ollama runtime (local only)
├── scripts/                 setup/start scripts, model + Ollama downloaders, parallel downloader
├── tests/                   unit, API and end-to-end tests
├── logs/                    vigil_x.log, ollama.log
├── .env.example · CHANGELOG.md · README.md
```

---

## Requirements

* Windows 10/11 (Linux/macOS also work with the equivalent commands)
* **Python 3.10 – 3.12** (3.12 recommended)
* **Node.js 20+** for the frontend (or a project-local Node via `pip install nodejs-wheel`)
* NVIDIA GPU recommended (8 GB is enough: ~1.5 GB backend + ~3.3 GB qwen3-vl:4b); CPU works, just slower
* FFmpeg: bundled via `imageio-ffmpeg`
* Disk: ~6 GB dependencies + models, +3.3 GB for the local VLM

---

## Windows installation

### 1. Backend

```powershell
cd backend
py -3.12 -m venv venv            # or: uv venv --seed venv --python 3.12
venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
pip install -r requirements.txt
cd ..
copy .env.example .env
```

Or in one step: `powershell -ExecutionPolicy Bypass -File scripts\setup_backend.ps1`

### 2. Models (detector, SigLIP 2, face models)

```powershell
backend\venv\Scripts\python scripts\download_models.py      # --skip-face to omit the face models
```

Stores `models/yolo11m.pt`, `models/siglip2-base-patch16-224/` and `models/face/` (YuNet + SFace ONNX, ~39 MB)
inside the project so VIGIL X runs offline.

### 3. Local AI (Ollama + Qwen3-VL) — optional but recommended

```powershell
backend\venv\Scripts\python scripts\setup_ollama.py      # portable Ollama → tools/ollama, qwen3-vl:4b → models/ollama (3.1 GB)
#   options: --model <name:tag>   --skip-binary   --skip-model   --connections N
```

Nothing else to configure: with `LLM_PROVIDER=auto` / `VLM_PROVIDER=auto` (default) VIGIL X starts
`tools/ollama/ollama.exe` itself (one model loaded at a time, flash attention, q8 KV cache), discovers installed
models through `/api/tags` + `/api/show` (capabilities `vision`, `thinking`) and stops it on exit. An Ollama you
installed system-wide on `127.0.0.1:11434` is used as-is. Other providers (OpenAI-compatible, Gemini, Claude) are
configured in `.env`; keys stay in `.env` and are never displayed.

### 4. Frontend

```powershell
cd frontend
npm install
npm run build        # production build served by the backend at http://localhost:8000
npm run dev          # OR hot-reload dev server at http://localhost:5173 (proxies /api to :8000)
```

---

## Environment variables (`.env`)

See `.env.example` for every option with comments. The most important ones:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` / `DATABASE_URL` | `./data` / `sqlite:///./data/vigil_x.db` | footage, derived data, database |
| `MODEL_PATH` · `DETECT_CLASSES` · `TRACKER` | `yolo11m.pt` · 9 classes · `bytetrack` | detection + tracking |
| `TRACKING_FPS` · `TARGET_FPS` (spec: `FRAME_SAMPLE_RATE`) | `5` · `1` | detector rate · index-frame rate |
| `EMBEDDING_MODEL` · `TOP_K` | SigLIP 2 base · `30` | semantic search |
| `LLM_PROVIDER` · `VLM_PROVIDER` | `auto` | `auto`/`ollama` (local), `openai`, `gemini`, `anthropic`, `local`, `none` |
| `OLLAMA_BASE_URL` · `OLLAMA_DEFAULT_MODEL` · `OLLAMA_AUTOSTART` | `http://127.0.0.1:11434` · `qwen3-vl:4b` · `true` | local AI |
| `VLM_MAX_CANDIDATES` (spec: `MAX_VLM_CANDIDATES`) · `VLM_TIMEOUT_SECONDS` | `6` · `120` | GPU budget per query |
| `LLM_PARSE_MODE` · `LLM_SUMMARIZE` | `fallback` · `true` | LLM only when the rule parser finds nothing; rephrase verified answers |
| `FACE_RECOGNITION_ENABLED` · `FACE_MATCH_THRESHOLD` · `FACE_POSSIBLE_THRESHOLD` · `FACE_MATCH_MARGIN` | `true` · `0.45` · `0.363` · `0.05` | consent-based recognition |
| `FACE_MIN_SIZE_PX` · `FACE_MIN_PERSON_PX` · `FACE_TRACK_MIN_MATCHES` | `40` · `160` · `2` | face quality gates + track smoothing |
| `FACE_STORE_UNKNOWN_EMBEDDINGS` | `true` | keep unknown-face embeddings for re-matching (purgeable) |
| `CONFIDENCE_WEIGHTS` · `EVIDENCE_CONFIRMED_THRESHOLD` · `EVIDENCE_LIKELY_THRESHOLD` | see file · `0.80` · `0.60` | Evidence Confidence engine |
| `CAMERA_TRAVEL_MIN_SECONDS` · `CAMERA_TRAVEL_MAX_SECONDS` | `0` · `600` | default travel window for unconfigured camera pairs |
| `ANOMALY_ELEVATED_Z` · `ANOMALY_UNUSUAL_Z` | `1.5` · `3.0` | activity baseline levels |

---

## Running

```powershell
backend\venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000** — API docs at `/docs`. Jury mode: **http://localhost:8000/present** (arrow keys,
*Full screen* button). The **VIGIL X** desktop shortcut runs the same app in its own window (`desktop/launcher.py`)
and stops the engine and the portable Ollama when the window closes.

On first start after an upgrade the database is migrated **additively**: a backup is written to `data/backups/`,
new tables and nullable columns are added, nothing is dropped or rewritten (see `schema_migrations`).

---

## Adding, scanning and processing videos

**Upload:** *Videos → Upload video* → choose camera, recording date, start time, timezone → *Upload & register*.
**Folder:** copy files into `data/videos/` → *Videos → Scan videos* → confirm → *Register & process*.

Processing shows real job states and progress (reading, playback copy, frames, detection, tracking, embeddings,
**face matching (VERIFYING)**, events, evidence, **search index (INDEXING)**, cross-camera). After each video, all
enabled watches are replayed on it.

> **Recording start time matters.** Absolute timestamps are `start + video offset`; fixing a start time later
> re-stamps everything derived from it.

---

## Searching footage

* *Show all people detected at the CTC entrance.* → *What about the staircase?* → *Only at night.* → *Show the second one.*
* *Did a red car pass through the main gate?* · *Find a person wearing a black shirt.* · *Show me people carrying bags.*
* *What happened between 10 and 10:30?* · *What changed at CAM-01 between 9:00 and 9:30?* · *How many people were at the entrance?*
* *Did the same person later appear at Lab 1?* · *Show unknown people near the staircase.*
* Enrolled participants: *Where was Demo Person A?* · *Which cameras saw Demo Person A?* · *Show Demo Person A's journey.*
  · *Did Demo Person A move from the entrance to Lab 1?* · *What happened before Demo Person A entered the lab?*

Every answer shows the evidence status and a toolbar:

| Tab | Content |
|---|---|
| **Evidence proof** | Evidence Confidence with its weighted components (detector, SigLIP, VLM, face identity, time filter, topology, coverage — renormalised over the signals that exist), coverage of cameras / requested time / evidence frames with **GREEN · STRONG / YELLOW · PARTIAL / RED · INSUFFICIENT**, missing footage intervals, supporting evidence, limitations |
| **Challenge answer** | Counter-evidence: VLM rejections, pixel-vs-SigLIP disagreement, lighting/size limits, topology violations, simultaneous sightings, conflicting faces, alternative continuity candidates, missing footage |
| **Evidence graph** | query → result → track / identity → detection → evidence frame → (VLM verdict) → video → camera |
| **Query understanding** | intent, identity, objects, colours, clothing, cameras, time, lighting, visual query, strategy |
| **Performance** | measured stage timings (parse, SQLite, SigLIP/FAISS, VLM, ProofGraph, LLM), VLM calls and cache hits |

Each result card adds **identity** (enrolled name + CONFIRMED/LIKELY, or UNKNOWN PERSON), **EC** (Evidence
Confidence), **“Why did this match?”**, feedback buttons and **Case** (add to an investigation).

### Evidence Confidence

`EC = Σ wᵢ·cᵢ / Σ wᵢ` over the components that exist for a result (weights from `CONFIDENCE_WEIGHTS`). A VLM “no”
counts as 0, “uncertain” as 0.35. It is labelled *Evidence Confidence* — a transparent summary of signals, not a
probability of truth or of identity. Status: **CONFIRMED** ≥ 0.80 with no contradicting evidence and adequate
coverage, **LIKELY** ≥ 0.60, **UNCERTAIN** otherwise, **INSUFFICIENT EVIDENCE** when nothing verified was found.

### Camera alias memory (clarify once)

Unknown places (*main gate*, *the doors*) trigger one question; the answer is stored in `camera_aliases` and reused
forever (including after restarts). Manage aliases in **Settings**.

### Cross-camera continuity and journeys

* **Anonymous tracks:** distinctiveness z-score, margin, mutual best match, colour signature and time gap
  (validated against the topology when configured) → VERIFIED (VLM agreed) / LIKELY / UNCERTAIN / NOT VERIFIED.
* **Enrolled participants:** face-matched sightings are merged into visits per camera; each transition is checked
  against the camera topology — **CONFIRMED** needs both sightings face-CONFIRMED and a configured travel window;
  unconfigured pairs are capped at LIKELY; overlaps (“seen on two cameras at once”) and too-fast/too-slow transitions
  make it UNCERTAIN and appear as counter-evidence. Configure travel times on **Cameras → Camera topology**.

---

## Consent-based face recognition

Only for adults who were **intentionally enrolled as demo participants**:

1. **Identities → Enroll**: display name, 1–10 front-facing photos (each previewed with its detected face box and
   quality), tick *adult (18+)* and *explicit consent*. Photos are analysed and discarded; only 128-number SFace
   embeddings are stored. Enrollment without both confirmations is refused by the API.
2. Existing footage is re-matched automatically. Videos processed **before** face recognition existed can be
   backfilled with **Scan processed footage for faces** (uses the saved frames and stored person detections).
3. Per face: **MATCH** (similarity ≥ `FACE_MATCH_THRESHOLD` and ≥ `FACE_MATCH_MARGIN` ahead of every other identity),
   **POSSIBLE** (weak, never named), otherwise **UNKNOWN PERSON**. Faces smaller than `FACE_MIN_SIZE_PX` or people
   shorter than `FACE_MIN_PERSON_PX` are not compared at all.
4. Per track: **CONFIRMED** needs ≥ `FACE_TRACK_MIN_MATCHES` matching frames that are ≥ 60 % of matched frames;
   one weak or single frame can never switch an identity; contradictory faces on one track → no identity.
5. **Delete** removes the identity, its embeddings, its face matches (embeddings of matched faces included) and every
   face-evidence image labelled with the name. *Pause recognition* keeps the data but stops matching.

> Faces must be large and front-facing. The bundled sample videos are 848×478 with people mostly < 160 px tall, so
> they yield no usable faces (correctly). Record identity demos with participants walking towards the camera.

---

## Watches, alerts, investigations, analytics

* **Alerts → New watch**: a question (same language as Search) + optional camera / time window + severity. It is
  replayed on recorded footage immediately and after every processed video; each verified result raises one alert
  (de-duplicated) linked to its evidence. OPEN → ACKNOWLEDGED → RESOLVED (or reopen). Always labelled RECORDED.
* **Investigations**: attach results with **Case**, add USER NOTEs, generate an **AI-GENERATED SUMMARY** (local LLM,
  grounding-checked), download **JSON / CSV / printable HTML** (use the browser's *Save as PDF*).
* **Analytics**: counts, busiest hour, peak people per frame, activity baselines (needs ≥ 4 covered 15-minute periods
  per camera), and **What changed?** (object counts + arrivals/departures + pixel change between the first and last
  indexed frame of a window).
* **Timeline**: footage coverage lanes + events + enrolled-participant sightings + alerts.
* **ProofGraph page**: every past query with its stored proof.

---

## API (summary)

```
GET  /api/health  /api/settings  /api/dashboard  /api/models/status  /api/system/metrics  /api/privacy
GET  /api/cameras  /api/cameras/health  /api/cameras/topology        POST/PUT/DELETE /api/cameras/topology[/{id}]
GET  /api/videos[/{id}]  POST /api/videos/upload|scan|register|process
GET  /api/evidence/{id}[/frame|/clip|/neighbors|/faces|/verifications]   POST /api/verify/evidence
POST /api/search  POST /api/chat  GET /api/queries  GET /api/queries/{id}/proof  GET /api/export
GET/POST /api/identities   POST /api/identities/preview|rematch|scan-footage   POST /api/identities/{id}/references
PUT/DELETE /api/identities/{id}   GET /api/identities/{id}/appearances|journey   GET /api/faces
POST /api/privacy/purge-unknown-faces
GET/POST /api/standing-queries   POST /api/standing-queries/{id}/run   GET /api/alerts   POST /api/alerts/{id}/acknowledge|resolve|reopen
GET/POST /api/investigations  …/{id}/items  …/{id}/notes  …/{id}/ai-summary  …/{id}/report?format=json|csv|html
POST/GET /api/feedback   GET /api/analytics  /api/anomalies  /api/what-changed  /api/timeline
```

---

## Testing

```powershell
backend\venv\Scripts\python scripts\fetch_sample_videos.py   # small public sample videos -> tests/fixtures
backend\venv\Scripts\python -m pytest                        # or: scripts\run_tests.ps1
```

Tests run in an isolated temporary data directory (your `data/` is never touched). Coverage includes ingestion,
timestamps, parsing, search, memory, aliases, low light, continuity, the end-to-end pipeline on real footage, plus
Round 2: migrations without data loss, Ollama client (mocked HTTP), VLM verdict validation, verification cache,
LLM grounding (including invented counts), face gallery decisions and track smoothing (synthetic embeddings only),
enrollment rules, deletion, unknown-person filtering, face backfill, topology rules, journeys, counter-evidence,
coverage, confidence weights, watches/alerts, investigations/reports, feedback, analytics, what-changed, timeline,
model status and honest behaviour when Ollama / the VLM / face models are unavailable.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Models page shows LOADING for LLM/VLM | Ollama is starting (first start also discovers the GPU); it becomes READY within ~30–60 s |
| First VLM check is slow | The model is loading into GPU memory (~10–20 s once); later checks take ~3–5 s and repeated checks are cached |
| No identity results | Faces too small / not visible, participant not enrolled, or recognition paused — see Identities → Face search |
| `No Python at '...AppData\Roaming\uv\python\...'` | `backend\venv\pyvenv.cfg` must say `home = D:\VigilX\backend\python312` |
| Slow downloads on throttled Wi-Fi | `python scripts/fast_download.py URL OUT` (parallel ranges) |
| Video stuck after a crash | On restart it is marked FAILED with a message; click Process (shown as RETRYING) |
| Restore the pre-upgrade database | Stop VIGIL X and copy the newest file from `data/backups/` over `data/vigil_x.db` |

Logs: `logs/vigil_x.log`, `logs/ollama.log`. Database tools: `python database/db_tools.py stats|integrity|backup|schema`.

---

## Hackathon demo (3 minutes)

1. **Dashboard → Jury mode** (`/present`): live counts, model states, the measured processing trace of the last video.
2. **Search:** *Did a red car pass through the main gate?* → clarify once → evidence → **Evidence proof** → **Challenge answer**.
3. *Show me people carrying bags.* → VLM-checked results; open a frame → face boxes, previous/next frame, VLM verdict.
4. *Show all people at the entrance* → *What about the staircase?* → *Show the second one.* → *What did the person look like?*
5. With a consenting enrolled participant: *Show Demo Person A's journey.* → topology-validated path; set a too-short
   travel window on **Cameras** and ask again → UNCERTAIN + counter-evidence.
6. **Alerts:** create a watch → alerts from recorded footage. **Case** a result → **Investigations** → report.

Record CAM-01 (entrance), CAM-02 (staircase), CAM-03 (lab) with fixed phone positions and consenting adult volunteers
walking the route towards the cameras; include a night/low-light clip.

---

## Privacy

* Recognition only of **adults who were intentionally enrolled with recorded consent**; no open-ended or public
  identification, no internet face sources, no cloud face recognition, children are not to be enrolled.
* Everyone else is **UNKNOWN PERSON**; weak similarities (POSSIBLE) are never shown as a name.
* No age, gender, ethnicity, emotion or intent inference anywhere.
* Reference photos are not stored — only embeddings in the local database; **Delete** removes them with all matches
  and labelled face evidence; unknown-face embeddings can be purged (Settings → Privacy).
* All processing is local; no video is uploaded; Ollama runs on this machine. LLM/VLM output is validated, never
  stored as fact, and cannot execute code or access files.

---

## Model licensing

| Component | Licence | Notes |
|---|---|---|
| Ultralytics YOLO11 | AGPL-3.0 | commercial use needs an Ultralytics licence |
| SigLIP 2 (google/siglip2-base-patch16-224) | Apache-2.0 | |
| YuNet face detector (OpenCV Zoo) | MIT | |
| SFace face recogniser (OpenCV Zoo) | Apache-2.0 | |
| Qwen3-VL 4B (via Ollama) | Apache-2.0 | see the Qwen model card |
| Ollama | MIT | |
| ByteTrack / BoT-SORT (via Ultralytics) | MIT / AGPL-3.0 (Ultralytics) | |
| FAISS | MIT | |
