# Holmes River Gauge Reader — Project Plan

## Overview

A SpyPoint FLEX-M trail camera is mounted at a bridge stanchion on the Holmes River, pointed at a marked gauge staff. Each gauge line is approximately 20 cm apart. This project automatically reads the river level from camera images and serves current and historical readings over HTTP.

**Pipeline:**
1. SpyPoint cloud API → latest image URL
2. Claude vision API → gauge reading (level in metres)
3. SQLite → stored reading
4. FastAPI → HTTP endpoints serving current and historical data

---

## Architecture

```
holmes-river/
├── .env                   # credentials and config (not committed)
├── .env.example           # template
├── requirements.txt
├── fetch_latest.py        # standalone script to test image download
├── main.py                # entry point: starts poller + web server
└── src/
    ├── config.py          # pydantic-settings: loads .env
    ├── spypoint.py        # SpyPoint cloud API client
    ├── gauge.py           # Claude vision gauge reading
    ├── db.py              # SQLite read/write helpers
    ├── api.py             # FastAPI web server
    └── poller.py          # APScheduler job: fetch → read → store
```

---

## Components

### `src/config.py`
Pydantic-settings model loading from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPYPOINT_USERNAME` | — | SpyPoint account email |
| `SPYPOINT_PASSWORD` | — | SpyPoint account password |
| `SPYPOINT_CAMERA_ID` | _(first camera)_ | Camera to poll; blank = use first on account |
| `ANTHROPIC_API_KEY` | — | Anthropic API key for Claude vision |
| `POLL_INTERVAL_MINUTES` | `30` | How often to check for new images |
| `DB_PATH` | `./readings.db` | SQLite database file path |
| `GAUGE_LINE_SPACING_CM` | `20` | Distance between gauge lines in cm |

---

### `src/spypoint.py`
SpyPoint REST API client using `httpx`.

**Base URL:** `https://restapi.spypoint.com/api/v3`

| Action | Method | Path |
|--------|--------|------|
| Login | `POST` | `/user/login` |
| List cameras | `GET` | `/camera/all` |
| Get photos | `POST` | `/photo/all` |

Auth returns a JWT Bearer token. Tokens are cached and automatically refreshed on 401.

Photo URL is constructed from the response as:
```
https://{photo.large.host}/{photo.large.path}
```

**Key methods:**
- `login()` — authenticate and store token
- `get_cameras()` → `list[Camera]`
- `get_latest_photo(camera_id)` → `Photo | None`
- `get_photos(camera_id, limit)` → `list[Photo]`

---

### `src/gauge.py`
Claude vision call using the `anthropic` SDK.

```python
def read_gauge(image_url: str, line_spacing_cm: int) -> GaugeReading
```

- Sends the image URL to `claude-sonnet-4-6`
- System prompt is marked as an ephemeral cache block (reduces cost on repeated calls)
- Returns a typed `GaugeReading` dataclass

**System prompt:**
>You are a river gauge reader. You will be shown trail camera images of a gauge staff on a river. The horizontal lines on the gauge are 20 cm apart. Identify the current water surface level by finding where the waterline meets the gauge staff, be careful to ignore any wet lines on the wall and actually look at the what is the river. The numbers are roughly evenly spaced apart, read the nearest visible number and infer how far the river is between the number below, and estimate the level in bridge units and also metres. Always respond with valid JSON only:
> `{"level_bridge": <float or null>, "level_m": <float or null>, "confidence": "high|medium|low", "notes": "<one sentence>"}`
> If the gauge is not visible or the image is too dark, set level_m to null.  



**`GaugeReading` fields:**
- `level_bridge: float | None` — raw number read from the gauge staff (as marked, e.g. `10.4`)
- `level_m: float | None` — water depth in metres derived from `level_bridge` and line spacing
- `confidence: str` — `"high"`, `"medium"`, or `"low"`
- `notes: str` — one-sentence description from the model
- `raw_json: str` — full model response for debugging

---

### `src/db.py`
SQLite via stdlib `sqlite3`.

**Schema:**
```sql
CREATE TABLE IF NOT EXISTS readings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,   -- ISO-8601 UTC
    level_m    REAL,            -- water level in metres (null if unreadable)
    confidence TEXT,
    notes      TEXT,
    image_url  TEXT,
    raw_json   TEXT             -- full Claude response
);
```

**Functions:**
- `init_db(path)` — create table if not exists
- `insert_reading(conn, reading)`
- `get_latest(conn)` → single row
- `get_history(conn, limit, since)` → list

---

### `src/api.py`
FastAPI application.

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check + latest reading summary |
| `GET /current` | Latest `GaugeReading` as JSON |
| `GET /history?limit=50&since=ISO_DATE` | Paginated historical readings |

---

### `src/poller.py`
APScheduler `BackgroundScheduler` running on `POLL_INTERVAL_MINUTES`.

**Each tick:**
1. Call `spypoint.get_latest_photo(camera_id)`
2. Compare `photo.taken_at` against the most recent stored reading — skip if already processed
3. Call `gauge.read_gauge(photo.url)`
4. Call `db.insert_reading()`
5. Log result

---

### `main.py`
Entry point:
1. `init_db()`
2. Start the APScheduler poller
3. Run `uvicorn` on port 8000 (blocking)

---

## Dependencies

```
anthropic>=0.40
fastapi
uvicorn[standard]
httpx
apscheduler
pydantic-settings
python-dotenv
```

Install into a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or with `uv`:
```bash
uv venv .venv
uv pip install -r requirements.txt
```

---

## Running

```bash
# Test image download only
.venv/bin/python fetch_latest.py --list-cameras
.venv/bin/python fetch_latest.py --out latest.jpg

# Start the full service (poller + web API)
.venv/bin/python main.py
```

Web API available at `http://localhost:8000`.

---

## Estimated API Costs

Using `claude-sonnet-4-6` (~$0.007 per image):

| Poll frequency | Readings/day | Cost/month |
|----------------|-------------|------------|
| Every 2 hours | 12 | ~$2.50 |
| Every hour | 24 | ~$5.00 |
| Every 30 min | 48 | ~$10.00 |

Switching to `claude-haiku-4-5` reduces this ~10×. Cost only accrues for genuinely new images — duplicate photos are skipped.

---

## Verification

```bash
# 1. Confirm image download works
.venv/bin/python fetch_latest.py

# 2. Start service and check endpoints
.venv/bin/python main.py &
curl http://localhost:8000/current
curl http://localhost:8000/history

# 3. Inspect the database directly
sqlite3 readings.db "SELECT * FROM readings ORDER BY timestamp DESC LIMIT 5;"
```
