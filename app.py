import os
import base64
import tempfile

from flask import Flask, request, jsonify, render_template_string
from llama_cpp import Llama
from chromadb import EmbeddingFunction, Documents, Embeddings

from preprocesser import TextPreprocessor
from chunker import Chunker
from dataloader import DataLoader
from vectordatabase import VectorDatabase
from highlighter import Highlighter

# ── Qwen3 Local Embedding Model ───────────────────────────────────────────────
_llm_model = Llama(
    model_path="/home/stark/Embedding_model_Qwen/Qwen3-Embedding-4B-Q5_K_M.gguf",
    embedding=True,
    n_ctx=512,
    verbose=False
)

class QwenEmbedding(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        return [_llm_model.embed(text) for text in input]

# ── App & global pipeline state ───────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

_preprocessor = TextPreprocessor()
_chunker      = Chunker(_preprocessor)
_loader       = DataLoader()
_highlighter  = Highlighter()
_vector_db    = VectorDatabase(QwenEmbedding())

_state = {
    "pdf_path": None,
    "pdf_name": None,
    "ingested": False,
}

# ── HTML Template ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Smart PDF Highlighter</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet"/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg:      #0d0d0d;
    --surface: #161616;
    --border:  #2a2a2a;
    --accent:  #f5e642;
    --text:    #f0ede6;
    --muted:   #666;
    --radius:  6px;
  }
  body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; min-height: 100vh; display: flex; flex-direction: column; }
  header { padding: 20px 40px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 16px; }
  header h1 { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem; letter-spacing: -0.02em; }
  .tag { background: var(--accent); color: #000; font-size: 0.62rem; font-weight: 700; padding: 3px 8px; border-radius: 2px; letter-spacing: 0.1em; text-transform: uppercase; }
  .model-badge { margin-left: auto; font-size: 0.7rem; color: var(--muted); border: 1px solid var(--border); padding: 4px 12px; border-radius: var(--radius); }
  .model-badge span { color: var(--accent); }
  .layout { display: grid; grid-template-columns: 360px 1fr; flex: 1; overflow: hidden; height: calc(100vh - 65px); }
  .panel { border-right: 1px solid var(--border); display: flex; flex-direction: column; overflow-y: auto; }
  .section { padding: 24px 28px; border-bottom: 1px solid var(--border); }
  .section-label { font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
  .upload-zone { border: 1.5px dashed var(--border); border-radius: var(--radius); padding: 28px 20px; text-align: center; cursor: pointer; transition: border-color 0.2s, background 0.2s; position: relative; }
  .upload-zone:hover, .upload-zone.drag-over { border-color: var(--accent); background: rgba(245,230,66,0.04); }
  .upload-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-icon { font-size: 1.8rem; margin-bottom: 8px; }
  .upload-zone p { font-size: 0.76rem; color: var(--muted); line-height: 1.6; }
  .filename { color: var(--accent); font-size: 0.76rem; margin-top: 8px; word-break: break-all; }
  .btn { width: 100%; padding: 11px; background: var(--accent); color: #000; border: none; border-radius: var(--radius); font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.82rem; cursor: pointer; transition: opacity 0.15s, transform 0.1s; margin-top: 12px; }
  .btn:hover:not(:disabled) { opacity: 0.85; transform: translateY(-1px); }
  .btn:disabled { opacity: 0.3; cursor: not-allowed; transform: none; }
  .btn.secondary { background: transparent; color: var(--text); border: 1px solid var(--border); }
  .btn.secondary:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .query-input { width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--text); font-family: 'DM Mono', monospace; font-size: 0.8rem; padding: 11px 13px; resize: vertical; min-height: 85px; line-height: 1.6; transition: border-color 0.2s; }
  .query-input:focus { outline: none; border-color: var(--accent); }
  .n-results-row { display: flex; align-items: center; gap: 10px; margin-top: 10px; font-size: 0.72rem; color: var(--muted); }
  .n-results-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .status { padding: 11px 14px; border-radius: var(--radius); font-size: 0.76rem; line-height: 1.5; display: none; margin-top: 10px; }
  .status.info    { background: rgba(245,230,66,0.07); border: 1px solid rgba(245,230,66,0.18); color: var(--accent); display: block; }
  .status.error   { background: rgba(255,80,80,0.07);  border: 1px solid rgba(255,80,80,0.2);  color: #ff7070; display: block; }
  .status.success { background: rgba(80,200,120,0.07); border: 1px solid rgba(80,200,120,0.18); color: #6fdc9a; display: block; }
  .spinner { display: inline-block; width: 11px; height: 11px; border: 2px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 6px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Viewer */
  .viewer { background: #181818; display: flex; flex-direction: column; align-items: center; overflow-y: auto; padding: 32px 20px; gap: 18px; }
  .viewer-placeholder { margin: auto; text-align: center; color: var(--muted); }
  .viewer-placeholder .big-icon { font-size: 3.5rem; margin-bottom: 14px; opacity: 0.25; }
  .viewer-placeholder p { font-size: 0.82rem; line-height: 1.9; }
  .pdf-canvas-wrap { box-shadow: 0 8px 48px rgba(0,0,0,0.7); border-radius: 2px; overflow: hidden; }
  canvas { display: block; }
  .page-label { font-size: 0.62rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase; align-self: flex-start; padding-left: 4px; }
  .viewer-controls { position: sticky; top: 0; width: 100%; display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(24,24,24,0.9); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); z-index: 10; font-size: 0.72rem; color: var(--muted); }
</style>
</head>
<body>

<header>
  <h1>Smart PDF Highlighter</h1>
  <span class="tag">Beta</span>
  <div class="model-badge">Embedding · <span>Qwen3-4B Local</span></div>
</header>

<div class="layout">

  <!-- Left Panel -->
  <div class="panel">
    <div class="section">
      <div class="section-label">01 — Upload PDF</div>
      <div class="upload-zone" id="uploadZone">
        <input type="file" id="fileInput" accept=".pdf"/>
        <div class="upload-icon">📄</div>
        <p>Drop your PDF here<br/>or click to browse</p>
        <div class="filename" id="filenameDisplay"></div>
      </div>
      <button class="btn" id="ingestBtn" disabled>Ingest PDF</button>
      <div class="status" id="ingestStatus"></div>
    </div>

    <div class="section">
      <div class="section-label">02 — Ask a Question</div>
      <textarea class="query-input" id="queryInput"
        placeholder="e.g. What is the refund policy?&#10;What are the key findings?"></textarea>
      <div class="n-results-row">
        <span>Chunks:</span>
        <input type="range" id="nResults" min="1" max="10" value="3"/>
        <span id="nResultsVal">3</span>
      </div>
      <button class="btn secondary" id="queryBtn" disabled>Highlight Answer</button>
      <div class="status" id="queryStatus"></div>
    </div>
  </div>

  <!-- Right Panel -->
  <div class="viewer" id="viewer">
    <div class="viewer-placeholder">
      <div class="big-icon">🔍</div>
      <p>Upload a PDF and ask a question.<br/>Relevant passages will be highlighted here.</p>
    </div>
  </div>

</div>

<script>
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

  let ingested = false;
  const fileInput      = document.getElementById('fileInput');
  const filenameDisplay= document.getElementById('filenameDisplay');
  const ingestBtn      = document.getElementById('ingestBtn');
  const ingestStatus   = document.getElementById('ingestStatus');
  const queryInput     = document.getElementById('queryInput');
  const queryBtn       = document.getElementById('queryBtn');
  const queryStatus    = document.getElementById('queryStatus');
  const nResultsSlider = document.getElementById('nResults');
  const nResultsVal    = document.getElementById('nResultsVal');
  const viewer         = document.getElementById('viewer');

  const setStatus = (el, msg, type) => { el.className = `status ${type}`; el.innerHTML = msg; };
  const clearStatus = el => { el.className = 'status'; };

  nResultsSlider.addEventListener('input', () => { nResultsVal.textContent = nResultsSlider.value; });

  fileInput.addEventListener('change', () => {
    const f = fileInput.files[0];
    if (!f) return;
    filenameDisplay.textContent = f.name;
    ingestBtn.disabled = false;
    ingested = false;
    queryBtn.disabled = true;
    clearStatus(ingestStatus);
    clearStatus(queryStatus);
  });

  const uploadZone = document.getElementById('uploadZone');
  uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f && f.type === 'application/pdf') {
      fileInput.files = e.dataTransfer.files;
      filenameDisplay.textContent = f.name;
      ingestBtn.disabled = false;
    }
  });

  ingestBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) return;
    ingestBtn.disabled = true;
    setStatus(ingestStatus, '<span class="spinner"></span> Embedding with Qwen3 locally... (may take a moment)', 'info');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res  = await fetch('/ingest', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) { setStatus(ingestStatus, '✗ ' + data.error, 'error'); }
      else {
        setStatus(ingestStatus, `✓ ${data.chunks} chunks · ${data.pages} pages ingested.`, 'success');
        ingested = true;
        queryBtn.disabled = false;
      }
    } catch (e) {
      setStatus(ingestStatus, '✗ Network error: ' + e.message, 'error');
    } finally {
      ingestBtn.disabled = false;
    }
  });

  queryBtn.addEventListener('click', async () => {
    const q = queryInput.value.trim();
    if (!q || !ingested) return;
    queryBtn.disabled = true;
    setStatus(queryStatus, '<span class="spinner"></span> Searching and highlighting...', 'info');
    try {
      const res  = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, n_results: parseInt(nResultsSlider.value) })
      });
      const data = await res.json();
      if (data.error)        { setStatus(queryStatus, '✗ ' + data.error, 'error'); }
      else if (!data.pdf_b64){ setStatus(queryStatus, '⚠ No relevant passages found.', 'error'); }
      else {
        setStatus(queryStatus, `✓ Highlighted ${data.matches} chunk(s).`, 'success');
        await renderPDF(data.pdf_b64);
      }
    } catch (e) {
      setStatus(queryStatus, '✗ Network error: ' + e.message, 'error');
    } finally {
      queryBtn.disabled = false;
    }
  });

  async function renderPDF(b64) {
    const binary = atob(b64);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

    const pdfDoc = await pdfjsLib.getDocument({ data: bytes }).promise;
    viewer.innerHTML = '';

    const bar = document.createElement('div');
    bar.className = 'viewer-controls';
    bar.innerHTML = `<span>Highlighted PDF</span><span>${pdfDoc.numPages} page${pdfDoc.numPages > 1 ? 's' : ''}</span>`;
    viewer.appendChild(bar);

    for (let p = 1; p <= pdfDoc.numPages; p++) {
      const page     = await pdfDoc.getPage(p);
      const scale    = Math.min(1.6, (viewer.clientWidth - 80) / page.getViewport({ scale: 1 }).width);
      const viewport = page.getViewport({ scale });

      const canvas   = document.createElement('canvas');
      canvas.width   = viewport.width;
      canvas.height  = viewport.height;
      await page.render({ canvasContext: canvas.getContext('2d'), viewport }).promise;

      const label = document.createElement('div');
      label.className = 'page-label';
      label.textContent = `Page ${p}`;

      const wrap = document.createElement('div');
      wrap.className = 'pdf-canvas-wrap';
      wrap.appendChild(canvas);

      viewer.appendChild(label);
      viewer.appendChild(wrap);
    }
    viewer.scrollTop = 0;
  }
</script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/ingest", methods=["POST"])
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    # Clean up previous temp file
    if _state["pdf_path"] and os.path.exists(_state["pdf_path"]):
        os.remove(_state["pdf_path"])

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    file.save(tmp.name)
    tmp.close()

    _state.update({"pdf_path": tmp.name, "pdf_name": file.filename, "ingested": False})

    try:
        pages_data = _loader.load_data(tmp.name)
        pages_data = _preprocessor.preprocess(pages_data)
        _highlighter.set_pages_data(pages_data)

        chunks = _chunker.chunk(pages_data)
        if not chunks:
            return jsonify({"error": "Could not extract text. Is this a scanned/image PDF?"}), 400

        _vector_db.create_collection("pdf_session")
        insert_chunks = chunks[1:] if len(chunks[0]["text"].split()) < 10 else chunks
        _vector_db.insert(insert_chunks)

        _state["ingested"] = True
        return jsonify({"message": "OK", "pages": len(pages_data), "chunks": len(insert_chunks)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
def query():
    if not _state["ingested"]:
        return jsonify({"error": "Please ingest a PDF first."}), 400

    body      = request.get_json() or {}
    query_txt = body.get("query", "").strip()
    n_results = int(body.get("n_results", 3))

    if not query_txt:
        return jsonify({"error": "Query cannot be empty."}), 400

    try:
        results = _vector_db.query(query_txt, n_results=n_results)
        matched = (results.get("metadatas") or [[]])[0]

        if not matched:
            return jsonify({"pdf_b64": None, "matches": 0})

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        out_tmp.close()
        _highlighter.highlight(_state["pdf_path"], results, out_tmp.name)

        with open(out_tmp.name, "rb") as f:
            pdf_bytes = f.read()
        os.remove(out_tmp.name)

        return jsonify({"pdf_b64": base64.b64encode(pdf_bytes).decode(), "matches": len(matched)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # threaded=False is important — llama_cpp is not thread-safe
    app.run(debug=True, port=5000, threaded=False)
