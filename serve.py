"""
Web viewer — single image + timeline graph + correction input.

Usage:
    python serve.py
    then open http://localhost:8000
"""

import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from src.config import settings
from src.db import init_db, get_all_photos, save_correction

app = FastAPI()
app.mount("/photos", StaticFiles(directory="photos"), name="photos")


def _get_photos():
    conn = init_db(settings.db_path)
    rows = get_all_photos(conn)
    conn.close()
    return [
        {
            "id": r["spypoint_id"],
            "taken_at": r["taken_at"][:16].replace("T", " "),
            "src": f"/photos/{Path(r['local_path']).name}" if r["local_path"] else "",
            "level": r["level"],
            "local_level": r["local_level"],
            "corrected_level": r["corrected_level"],
            "confidence": r["confidence"] or "",
            "notes": r["notes"] or "",
        }
        for r in rows
    ]


class CorrectionIn(BaseModel):
    spypoint_id: str
    corrected_level: Optional[float]


@app.post("/correct")
def correct(body: CorrectionIn):
    conn = init_db(settings.db_path)
    save_correction(conn, body.spypoint_id, body.corrected_level)
    conn.close()
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
def index():
    photos = _get_photos()
    labeled = sum(1 for p in photos if p["corrected_level"] is not None)
    data_json = json.dumps(photos)

    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Holmes River Gauge</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ height: 100%; background: #111827; color: #e5e7eb; font-family: system-ui, sans-serif; }}
  body {{ display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}

  header {{
    padding: 0.5rem 1.2rem;
    background: #1f2937; border-bottom: 1px solid #374151;
    display: flex; align-items: center; gap: 1.5rem; flex-shrink: 0;
  }}
  header h1 {{ font-size: 1.1rem; font-weight: 600; }}
  header span {{ font-size: 0.8rem; color: #9ca3af; }}

  .info-bar {{
    display: flex; align-items: center; gap: 1rem;
    padding: 0.45rem 1.2rem; background: #1f2937;
    border-bottom: 1px solid #374151; flex-shrink: 0;
  }}
  .level      {{ font-size: 2rem; font-weight: 700; min-width: 3.5rem; }}
  .corrected  {{ font-size: 0.75rem; color: #9ca3af; white-space: nowrap; }}
  .date       {{ font-size: 0.8rem; color: #9ca3af; white-space: nowrap; }}
  .notes      {{ font-size: 0.78rem; color: #d1d5db; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .conf       {{ font-size: 0.7rem; font-weight: 600; text-transform: uppercase; white-space: nowrap; }}

  .correct-form {{
    display: flex; align-items: center; gap: 0.4rem; margin-left: auto; flex-shrink: 0;
  }}
  .correct-form label {{ font-size: 0.75rem; color: #9ca3af; white-space: nowrap; }}
  .correct-form input {{
    width: 5rem; padding: 0.25rem 0.4rem;
    background: #374151; border: 1px solid #4b5563; border-radius: 4px;
    color: #e5e7eb; font-size: 0.9rem; text-align: center;
  }}
  .correct-form button {{
    padding: 0.25rem 0.7rem; border-radius: 4px; border: none;
    background: #3b82f6; color: #fff; font-size: 0.8rem; cursor: pointer;
  }}
  .correct-form button:hover {{ background: #2563eb; }}
  #save-status {{ font-size: 0.75rem; color: #34d399; min-width: 3rem; }}

  .photo-wrap {{
    flex: 1 1 auto; min-height: 0;
    display: flex; align-items: center; justify-content: center;
    background: #000; overflow: hidden; position: relative;
  }}
  .photo-wrap img {{ max-width: 100%; max-height: 100%; object-fit: contain; display: block; }}
  .nav-btn {{
    position: absolute; top: 50%; transform: translateY(-50%);
    background: rgba(0,0,0,0.55); border: 1px solid #4b5563;
    color: #fff; font-size: 1.4rem; padding: 0.5rem 0.8rem;
    cursor: pointer; border-radius: 6px; user-select: none;
  }}
  .nav-btn:hover {{ background: rgba(0,0,0,0.85); }}
  #btn-prev {{ left: 0.75rem; }}
  #btn-next {{ right: 0.75rem; }}

  .chart-wrap {{
    height: 200px; flex-shrink: 0;
    padding: 0.5rem 1rem 0.75rem;
    background: #1f2937; border-top: 1px solid #374151;
  }}
</style>
</head>
<body>

<header>
  <h1>Holmes River Gauge</h1>
  <span id="counter"></span>
  <span id="labeled-count">{labeled} labelled</span>
</header>

<div class="info-bar">
  <span class="level" id="level-val">—</span>
  <div style="display:flex;flex-direction:column;gap:0.1rem;">
    <span class="corrected" id="corrected-val"></span>
    <span class="conf" id="conf-val"></span>
  </div>
  <span class="date"  id="date-val"></span>
  <span class="notes" id="notes-val"></span>
  <span style="font-size:0.78rem;white-space:nowrap;" id="local-val"></span>
  <div class="correct-form">
    <label for="correction-input">Correct level:</label>
    <input id="correction-input" type="number" step="0.1" placeholder="e.g. 8.2">
    <button onclick="saveCorrection()">Save</button>
    <span id="save-status"></span>
  </div>
</div>

<div class="photo-wrap">
  <img id="main-photo" src="" alt="">
  <button class="nav-btn" id="btn-prev">&#8592;</button>
  <button class="nav-btn" id="btn-next">&#8594;</button>
</div>

<div class="chart-wrap">
  <canvas id="chart"></canvas>
</div>

<script>
const PHOTOS = {data_json};
let current = 0;

const CONF_COLOR = {{
  high:   '#34d399',
  medium: '#fbbf24',
  low:    '#f87171',
  '':     '#6b7280',
}};

function showPhoto(i) {{
  if (i < 0 || i >= PHOTOS.length) return;
  current = i;
  const p = PHOTOS[i];

  document.getElementById('main-photo').src = p.src;
  document.getElementById('date-val').textContent  = p.taken_at;
  document.getElementById('notes-val').textContent = p.notes;
  document.getElementById('conf-val').textContent  = p.confidence;
  document.getElementById('counter').textContent   = `${{i + 1}} / ${{PHOTOS.length}}`;
  document.getElementById('save-status').textContent = '';

  const displayLevel = p.corrected_level ?? p.level;
  document.getElementById('level-val').textContent = displayLevel != null ? displayLevel.toFixed(1) : '—';
  document.getElementById('level-val').style.color = p.corrected_level != null ? '#a78bfa' : (CONF_COLOR[p.confidence] || '#6b7280');
  document.getElementById('conf-val').style.color  = CONF_COLOR[p.confidence] || '#6b7280';

  if (p.corrected_level != null) {{
    document.getElementById('corrected-val').textContent = `✓ corrected`;
    document.getElementById('corrected-val').style.color = '#a78bfa';
  }} else {{
    document.getElementById('corrected-val').textContent = '';
  }}

  // Local model comparison
  const localEl = document.getElementById('local-val');
  if (p.local_level != null) {{
    const diff = p.corrected_level != null ? (p.local_level - p.corrected_level).toFixed(1) : null;
    const diffStr = diff != null ? ` (${{diff > 0 ? '+' : ''}}${{diff}})` : '';
    localEl.textContent = `model: ${{p.local_level.toFixed(1)}}${{diffStr}}`;
    localEl.style.color = diff != null && Math.abs(diff) <= 0.5 ? '#34d399' : '#fbbf24';
  }} else {{
    localEl.textContent = '';
  }}

  // Pre-fill correction input with existing corrected value (or estimated)
  document.getElementById('correction-input').value =
    p.corrected_level != null ? p.corrected_level : (p.level != null ? p.level.toFixed(1) : '');

  // Update chart highlight
  const y = p.corrected_level ?? p.level;
  chart.data.datasets[2].data = y != null ? [{{x: i, y}}] : [];
  chart.update('none');
}}

async function saveCorrection() {{
  const p = PHOTOS[current];
  const val = parseFloat(document.getElementById('correction-input').value);
  if (isNaN(val)) return;
  const resp = await fetch('/correct', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{spypoint_id: p.id, corrected_level: val}}),
  }});
  if (resp.ok) {{
    p.corrected_level = val;
    document.getElementById('save-status').textContent = '✓ saved';
    document.getElementById('labeled-count').textContent =
      PHOTOS.filter(p => p.corrected_level != null).length + ' labelled';
    showPhoto(current);
    // Update correction series on chart
    const corrPoints = PHOTOS
      .map((p, i) => p.corrected_level != null ? {{x: i, y: p.corrected_level}} : null)
      .filter(Boolean);
    chart.data.datasets[1].data = corrPoints;
    chart.update('none');
  }}
}}

document.getElementById('btn-prev').onclick = () => showPhoto(current - 1);
document.getElementById('btn-next').onclick = () => showPhoto(current + 1);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft')  showPhoto(current - 1);
  if (e.key === 'ArrowRight') showPhoto(current + 1);
  if (e.key === 'Enter') saveCorrection();
}});

// Chart datasets: estimated line, corrections scatter, selected highlight
const estPoints = PHOTOS
  .map((p, i) => p.level != null ? {{x: i, y: p.level, conf: p.confidence}} : null)
  .filter(Boolean);
const corrPoints = PHOTOS
  .map((p, i) => p.corrected_level != null ? {{x: i, y: p.corrected_level}} : null)
  .filter(Boolean);
const localPoints = PHOTOS
  .map((p, i) => p.local_level != null ? {{x: i, y: p.local_level}} : null)
  .filter(Boolean);

const chart = new Chart(document.getElementById('chart'), {{
  type: 'scatter',
  data: {{
    datasets: [
      {{
        label: 'Estimated',
        data: estPoints,
        showLine: true,
        borderColor: '#3b82f6',
        borderWidth: 1.5,
        pointRadius: 3,
        pointBackgroundColor: estPoints.map(p => CONF_COLOR[p.conf] || '#6b7280'),
        pointBorderColor: 'transparent',
        tension: 0.2,
        fill: false,
      }},
      {{
        label: 'Corrected',
        data: corrPoints,
        showLine: false,
        pointRadius: 6,
        pointBackgroundColor: '#a78bfa',
        pointBorderColor: '#fff',
        pointBorderWidth: 1.5,
      }},
      {{
        label: 'Local model',
        data: localPoints,
        showLine: true,
        borderColor: '#f97316',
        borderWidth: 1.5,
        borderDash: [4, 3],
        pointRadius: 3,
        pointBackgroundColor: '#f97316',
        pointBorderColor: 'transparent',
        tension: 0.2,
        fill: false,
      }},
      {{
        label: 'Selected',
        data: [],
        showLine: false,
        pointRadius: 11,
        pointBackgroundColor: 'transparent',
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
      }}
    ]
  }},
  options: {{
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        callbacks: {{
          label: ctx => {{
            const p = PHOTOS[ctx.raw.x];
            const corr = p.corrected_level != null ? `  ✓ ${{p.corrected_level.toFixed(1)}}` : '';
            return `${{p.taken_at}}  est=${{p.level?.toFixed(1) ?? '—'}}${{corr}}`;
          }}
        }}
      }}
    }},
    scales: {{
      x: {{
        type: 'linear',
        ticks: {{
          color: '#9ca3af',
          maxTicksLimit: 10,
          callback: val => {{
            const p = PHOTOS[Math.round(val)];
            return p ? p.taken_at.slice(0, 10) : '';
          }},
        }},
        grid: {{ color: '#374151' }},
      }},
      y: {{
        ticks: {{ color: '#9ca3af' }},
        grid: {{ color: '#374151' }},
        title: {{ display: true, text: 'Gauge level', color: '#9ca3af' }},
      }}
    }},
    onClick: (e, elements) => {{
      if (elements.length > 0) showPhoto(elements[0].element.$context.raw.x);
    }}
  }}
}});

showPhoto(0);
</script>
</body>
</html>""")


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
