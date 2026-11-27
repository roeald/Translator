import gc
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings

import torch
from flask import Flask, jsonify, render_template_string, request, send_file
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class _HideLengthWarning(logging.Filter):
    """Blendet nur den Hinweis zu max_new_tokens/max_length aus, andere Warnungen bleiben sichtbar."""

    def filter(self, record):
        return "Both `max_new_tokens`" not in record.getMessage()


logging.getLogger("transformers.generation.utils").addFilter(_HideLengthWarning())
# EasyOCR meldet auf Rechnern ohne Grafikkarte bei jedem Bild dasselbe
warnings.filterwarnings("ignore", message=".*pin_memory.*")

PORT = 5000
USE_PDF2ZH = False  # False = eigenes Layout (layout.py) mit Wortumbruch; True = altes pdf2zh (für Formeln)
NUM_BEAMS = 4       # höher = etwas bessere Übersetzung, aber langsamer
app = Flask(__name__)
# Auswählbare Modelle (Umschalten über die Oberfläche)
MODELS = [
    {"id": "facebook/nllb-200-distilled-600M", "label": "NLLB 600M - fast", "size": "~2.4 GB"},
    {"id": "facebook/nllb-200-distilled-1.3B", "label": "NLLB 1.3B - usually better, slower", "size": "~5.5 GB"},
    {"id": "facebook/nllb-200-3.3B", "label": "NLLB 3.3B - largest, very slow, needs lots of RAM", "size": "~17.6 GB"},
]
MODEL_IDS = [m["id"] for m in MODELS]
CHOICE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_choice.txt")


def saved_model():
    """Zuletzt gewähltes Modell (Standard: das kleine)."""
    try:
        with open(CHOICE_FILE, encoding="utf-8") as f:
            name = f.read().strip()
        return name if name in MODEL_IDS else MODEL_IDS[0]
    except OSError:
        return MODEL_IDS[0]


MODEL_NAME = saved_model()

model = None
tokenizers = {}  # ein Tokenizer pro Quellsprache (wird gecacht)
engine_lock = threading.RLock()
busy_lock = threading.Lock()  # Übersetzen und Modellwechsel nie gleichzeitig
current_src_lang = "deu_Latn"
current_tgt_lang = "hrv_Latn"
kept_original = []  # Stellen, die das Modell nicht brauchbar übersetzen konnte

LANGUAGES = [
    {"code": "deu_Latn", "iso": "de", "name": "German (Deutsch)", "size": "Included (0 MB extra)"},
    {"code": "hrv_Latn", "iso": "hr", "name": "Croatian (Hrvatski)", "size": "Included (0 MB extra)"},
    {"code": "eng_Latn", "iso": "en", "name": "English", "size": "Included (0 MB extra)"},
    {"code": "nld_Latn", "iso": "nl", "name": "Dutch (Nederlands)", "size": "Included (0 MB extra)"},
    {"code": "pol_Latn", "iso": "pl", "name": "Polish (Polski)", "size": "Included (0 MB extra)"},
    {"code": "rus_Cyrl", "iso": "ru", "name": "Russian (Русский)", "size": "Included (0 MB extra)"},
    {"code": "tur_Latn", "iso": "tr", "name": "Turkish (Türkçe)", "size": "Included (0 MB extra)"},
    {"code": "jpn_Jpan", "iso": "ja", "name": "Japanese (日本語)", "size": "Included (0 MB extra)"},
    {"code": "zho_Hans", "iso": "zh", "name": "Chinese Simplified (简体中文)", "size": "Included (0 MB extra)"},
]

ISO_MAP = {lang["code"]: lang["iso"] for lang in LANGUAGES}
NO_SPACE_LANGS = {"jpn_Jpan", "zho_Hans"}  # Zielsprachen ohne Leerzeichen zwischen Sätzen

# Satzende + Leerzeichen + Großbuchstabe; nicht nach Abkürzungen ("z. B.") oder Zahlen ("3. Mai")
SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])(?<!\b\w\.)(?<!\d\.)(?<!bzw\.)(?<!ca\.)(?<!inkl\.)(?<!ggf\.)(?<!Nr\.)(?<!Dr\.)"
    r"\s+(?=[A-ZÄÖÜА-Я\[(„\"])|(?<=;)\s+|(?<=[。！？])")
# Nummern am Anfang ("1. ", "2.1 ", "1 Einleitung", "A. ") bleiben unübersetzt
LIST_PREFIX = re.compile(r"^\s*((?:(?:\d+(?:\.\d+)*\.|\d+(?:\.\d+)+|[A-Z]\.|\d{1,2}(?=\s+[A-ZÄÖÜ]))\s+)+)")
# Punktlinien, Unterstriche usw. dürfen nie ins Modell: sie lösen Endlos-Wiederholungen aus
LEADERS = re.compile(r"((?:\.[ \t]?){4,}|…{2,}|(?:·[ \t]?){4,}|_{4,}|-{5,}|={5,})")
PLACEHOLDER = re.compile(r"(\{\s*v\s*\d+\s*\})", re.IGNORECASE)  # pdf2zh-Platzhalter {v0}
MAX_PIECE_CHARS = 400
# Wie die pdf2zh-Vorgabe, aber OHNE ".*Ital": sonst gelten kursive Wörter als Formel und bleiben unübersetzt
FORMULA_FONT_REGEX = (r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary"
                      r"|.*Mono|.*Code|.*Sym|.*Math)")


def is_model_cached(name=None):
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    hub = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    snapshots = os.path.join(hub, "models--" + (name or MODEL_NAME).replace("/", "--"), "snapshots")
    if not os.path.isdir(snapshots):
        return False
    # Nur "installiert", wenn die Gewichte vollständig da sind (nicht nur Ordner oder Konfiguration)
    for snapshot in os.listdir(snapshots):
        folder = os.path.join(snapshots, snapshot)
        if any(os.path.exists(os.path.join(folder, f)) for f in ("model.safetensors", "pytorch_model.bin")):
            return True
        for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            try:
                with open(os.path.join(folder, index), encoding="utf-8") as f:
                    shards = set(json.load(f)["weight_map"].values())
            except (OSError, ValueError, KeyError):
                continue
            if all(os.path.exists(os.path.join(folder, shard)) for shard in shards):
                return True
    return False


def switch_model(name):
    """Altes Modell entladen, neues laden. Bei einem Fehler bleibt die alte Auswahl bestehen."""
    global model, MODEL_NAME
    with engine_lock:
        previous = MODEL_NAME
        model = None  # erst entladen, damit nie zwei Modelle im Arbeitsspeicher liegen
        tokenizers.clear()
        gc.collect()
        MODEL_NAME = name
        try:
            load_engine("deu_Latn")
        except Exception:
            model = None
            tokenizers.clear()
            gc.collect()
            MODEL_NAME = previous  # wird bei der nächsten Übersetzung wieder geladen
            raise
        try:
            with open(CHOICE_FILE, "w", encoding="utf-8") as f:
                f.write(name)
        except OSError as e:
            print(f"[!] Model choice could not be saved: {e}")


def load_engine(src_lang):
    """Lädt das Modell nur einmal und cached den Tokenizer pro Quellsprache."""
    global model
    with engine_lock:
        if model is None:
            print(f"[+] Loading {MODEL_NAME} into RAM...")
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            model.eval()
            # Die Länge wird pro Aufruf über max_new_tokens bestimmt (Vorgabe max_length=200 entfernen)
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.max_length = None
        if src_lang not in tokenizers:
            tokenizers[src_lang] = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=src_lang)
        return tokenizers[src_lang]


def split_sentences(text):
    parts = [p.strip() for p in SENTENCE_SPLIT.split(text) if p and p.strip()]
    pieces = []
    for part in parts or [text]:
        while len(part) > MAX_PIECE_CHARS:  # überlange "Sätze" an Wortgrenzen teilen
            cut = part.rfind(" ", 0, MAX_PIECE_CHARS)
            cut = cut if cut > 0 else MAX_PIECE_CHARS
            pieces.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            pieces.append(part)
    return pieces


def generate(tok, pieces, beams):
    inputs = tok(pieces, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(current_tgt_lang),
            # Ausgabelänge an die Eingabe koppeln, damit eine Schleife schnell endet
            max_new_tokens=min(512, int(inputs["input_ids"].shape[1] * 2) + 20),
            num_beams=beams,
            no_repeat_ngram_size=4,  # gegen "stavkom 2. stavkom 2. ..."
        )
    return tok.batch_decode(generated, skip_special_tokens=True)  # -> Liste von Strings!


def looks_broken(source, result):
    """Erkennt erfundene Ausgaben des Modells."""
    if not result.strip() or len(result) > 2.2 * len(source) + 30:
        return True
    # Ziffern, die es im Original nicht gibt ("... Artikel 3 Absatz 1 ...")
    if len(re.findall(r"\d", result)) > len(re.findall(r"\d", source)):
        return True
    words = re.findall(r"[^\W\d_]+", result.lower())
    if any(words[i] == words[i + 1] == words[i + 2] for i in range(len(words) - 2)):
        return True
    grams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    src_words = re.findall(r"[^\W\d_]+", source.lower())
    src_grams = [tuple(src_words[i:i + 2]) for i in range(len(src_words) - 1)]
    most = max((grams.count(g) for g in set(grams)), default=0)
    return most >= 3 and most > max((src_grams.count(g) for g in set(src_grams)), default=0) + 1


def retry_translation(tok, piece):
    """Weitere Versuche für ein Stück. Gibt None zurück, wenn alles Unsinn bleibt."""
    ended = piece.rstrip().endswith((".", "!", "?", ":", ";", "。"))
    variants = [(piece if ended else piece + ".", NUM_BEAMS), (piece, 1), (piece, NUM_BEAMS + 2)]
    for text, beams in variants:
        out = generate(tok, [text], beams)[0]
        if not ended:
            out = out.rstrip(" .。")
        if not looks_broken(piece, out):
            return out
    clauses = [c for c in re.split(r"(?<=[,;:])\s+", piece) if c]  # lange Sätze in Teilsätze
    if len(clauses) > 1:
        outs = generate(tok, clauses, NUM_BEAMS)
        if not any(looks_broken(c, o) for c, o in zip(clauses, outs)):
            return " ".join(outs)
    return None


def run_model(text):
    """Ein Stück normalen Text mit NLLB übersetzen (Satz für Satz, mit Kontrolle)."""
    print(f"[Translating {len(text)} chars]: {text!r}")
    with engine_lock:
        tok = load_engine(current_src_lang)
        pieces = split_sentences(text)  # NLLB ist für einzelne Sätze gebaut
        decoded = []
        for i in range(0, len(pieces), 4):  # kleine Stapel schonen den Arbeitsspeicher
            batch = pieces[i:i + 4]
            for piece, out in zip(batch, generate(tok, batch, NUM_BEAMS)):
                if looks_broken(piece, out):
                    print(f"[!] Nonsense output, retrying: {piece!r}")
                    out = retry_translation(tok, piece)
                    if out is None:  # lieber das Original zeigen als Erfundenes
                        print(f"[!] Kept original text: {piece!r}")
                        kept_original.append(piece)
                        out = piece
                decoded.append(out)
    sep = "" if current_tgt_lang in NO_SPACE_LANGS else " "
    return sep.join(decoded)


def translate_plain(text):
    """Text ohne Punktlinien: Randleerzeichen und Listennummern bewahren, Rest übersetzen."""
    core = text.strip()
    if sum(c.isalpha() for c in core) < 2:
        return text  # Zahlen, Satzzeichen, einzelne Zeichen
    lead = text[:len(text) - len(text.lstrip())]
    tail = text[len(text.rstrip()):]
    # Listennummern nicht mitübersetzen, sonst wird aus "3." das Wort "Treća"
    prefix = ""
    m = LIST_PREFIX.match(core)
    if m and core[m.end():].strip():
        prefix = " ".join(m.group(1).split()) + " "
        core = core[m.end():]
    return lead + prefix + run_model(core) + tail


def translate_chunk(text):
    if not text or not text.strip():
        return text
    parts = LEADERS.split(text)
    if len(parts) > 1:  # ungerade Positionen sind die Punktlinien selbst
        return "".join(p[:40] if i % 2 else translate_plain(p) for i, p in enumerate(parts))
    wanted = [re.sub(r"\s", "", p).lower() for p in PLACEHOLDER.findall(text)]
    if not wanted:
        return translate_plain(text)
    result = translate_plain(text)
    got = [re.sub(r"\s", "", p).lower() for p in PLACEHOLDER.findall(result)]
    if sorted(got) == sorted(wanted):
        return result
    # Modell hat Platzhalter verloren/verändert: Teile dazwischen einzeln übersetzen
    parts = PLACEHOLDER.split(text)
    return "".join(p if i % 2 else translate_plain(p) for i, p in enumerate(parts))


def extract_source_text(prompt):
    """pdf2zh verpackt den zu übersetzenden Text je nach Version in einen anderen Prompt.
    Hier wird nur der eigentliche Text herausgeschnitten."""
    text = prompt
    if "Source Text:" in text:
        text = text.split("Source Text:", 1)[1]
        text = text.split("Translated Text:", 1)[0]
    elif "Input:\n\n" in text:
        text = text.split("Input:\n\n", 1)[1]
    return text.strip()


@app.route("/v1/chat/completions", methods=["POST"])
def openai_bridge():
    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):  # OpenAI-Format mit Teilen
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            user_text = content
            break

    source = extract_source_text(user_text)
    try:
        translated = translate_chunk(source)
    except Exception:
        # Niemals einen Fehler zurückgeben: pdf2zh wiederholt die Anfrage sonst endlos
        traceback.print_exc()
        translated = source
    return jsonify({
        "id": "offline-nllb",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", "nllb-200"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": translated},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    })


# r"""...""" (raw string): sonst meldet Python 3.12+ eine SyntaxWarning wegen "\." im JavaScript
HTML_UI = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Offline Layout-Preserving PDF Translator</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f4f6f9; color: #1f2937; margin: 0; padding: 30px; }
  .container { max-width: 780px; margin: auto; background: #fff; padding: 28px; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.08); }
  h1 { margin-top: 0; font-size: 22px; display: flex; align-items: center; justify-content: space-between; }
  .badge { font-size: 13px; padding: 5px 10px; border-radius: 20px; font-weight: 600; }
  .installed { background: #dcfce7; color: #166534; }
  .not-installed { background: #fef3c7; color: #92400e; }
  .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0; }
  label { font-weight: 600; font-size: 13px; display: block; margin-bottom: 6px; }
  select, input[type="file"] { width: 100%; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 14px; box-sizing: border-box; }
  .model-row { display: grid; grid-template-columns: 1fr auto; gap: 16px; align-items: end; margin: 16px 0; }
  .model-row button { width: auto; padding: 10px 18px; font-size: 14px; }
  .upload-box { border: 2px dashed #9ca3af; padding: 24px; text-align: center; border-radius: 8px; background: #f9fafb; margin-bottom: 16px; }
  button { background: #2563eb; color: white; border: none; padding: 14px 20px; font-size: 15px; font-weight: 600; border-radius: 8px; cursor: pointer; width: 100%; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #93c5fd; cursor: not-allowed; }
  .dl-grid { display: none; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }
  .btn-mono { background: #059669; }
  .btn-mono:hover { background: #047857; }
  .btn-dual { background: #4f46e5; }
  .btn-dual:hover { background: #4338ca; }
  .status { margin-top: 14px; font-size: 14px; font-weight: 500; color: #374151; text-align: center; }
  .info-box { background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px; font-size: 13px; margin-bottom: 16px; border-radius: 4px; }
</style>
</head>
<body>
<div class="container">
  <h1>
    Offline PDF Translator (Full Layout)
    <span id="modelBadge" class="badge not-installed">Checking model...</span>
  </h1>

  <div class="info-box">
    <strong>Layout engine + NLLB-200:</strong> Preserves tables, columns, images, and Croatian diacritics (č, ć, đ, š, ž) offline across all 9 languages. Check the black terminal window for live progress!
  </div>

  <div class="model-row">
    <div>
      <label for="modelSelect">Translation Model:</label>
      <select id="modelSelect" onchange="updateSwitchButton()"></select>
    </div>
    <button id="switchBtn" onclick="switchModel()" disabled>Switch Model</button>
  </div>

  <div id="installSection" style="display:none; margin-bottom: 18px;">
    <button id="installBtn" onclick="installModel()">Download & Install Local Language Model</button>
  </div>

  <div class="controls">
    <div>
      <label for="srcLang">Translate From:</label>
      <select id="srcLang"></select>
    </div>
    <div>
      <label for="tgtLang">Translate To:</label>
      <select id="tgtLang"></select>
    </div>
  </div>

  <div class="upload-box">
    <label for="pdfFile">Select PDF Document (100% processed locally on your PC)</label>
    <input type="file" id="pdfFile" accept=".pdf">
  </div>

  <label style="font-weight:500; margin: 0 0 14px 2px; cursor:pointer;">
    <input type="checkbox" id="ocrBox" checked style="width:auto; margin-right:6px;">
    Also translate text inside images (OCR) &mdash; slower; first OCR use downloads ~100 MB of models
  </label>

  <button id="translateBtn" onclick="translatePDF()">Translate PDF & Keep Formatting</button>
  <div id="status" class="status"></div>

  <div id="downloadSection" class="dl-grid">
    <button class="btn-mono" onclick="downloadResult('mono')">Download Translated PDF</button>
    <button class="btn-dual" onclick="downloadResult('dual')">Download Bilingual (Side-by-Side) PDF</button>
  </div>
</div>

<script>
const languages = {{ languages | tojson }};
const models = {{ models | tojson }};
let jobId = null;
let currentModel = null;
let installedModels = {};
let working = false;

function updateSwitchButton() {
  const select = document.getElementById('modelSelect');
  document.getElementById('switchBtn').disabled = working || !currentModel || select.value === currentModel;
}

function setWorking(state) {
  working = state;
  document.getElementById('translateBtn').disabled = state;
  document.getElementById('installBtn').disabled = state;
  document.getElementById('modelSelect').disabled = state;
  updateSwitchButton();
}

function fillModels() {
  const select = document.getElementById('modelSelect');
  const chosen = select.value;
  select.innerHTML = "";
  models.forEach(m => {
    const state = installedModels[m.id] ? "installed" : "download " + m.size;
    select.add(new Option(`${m.label} [${state}]`, m.id));
  });
  select.value = chosen || currentModel;
}

async function switchModel() {
  const select = document.getElementById('modelSelect');
  const status = document.getElementById('status');
  const target = models.find(m => m.id === select.value);
  if (!installedModels[target.id] &&
      !confirm(`${target.label} has to be downloaded first (${target.size}, internet required). Continue?`)) return;
  setWorking(true);
  status.textContent = "Switching model... A download can take a long time. Watch the black terminal window.";
  try {
    const body = new FormData();
    body.append("model", target.id);
    const res = await fetch('/model', { method: 'POST', body });
    const data = await res.json();
    status.textContent = data.message;
  } catch (e) {
    status.textContent = "Lost connection to the local server (possibly not enough RAM). Check the black terminal window.";
  }
  setWorking(false);
  await checkStatus();
  select.value = currentModel;
  updateSwitchButton();
}

function populateDropdowns() {
  const src = document.getElementById('srcLang');
  const tgt = document.getElementById('tgtLang');
  languages.forEach(l => {
    const label = `${l.name} — [${l.size}]`;
    src.add(new Option(label, l.code));
    tgt.add(new Option(label, l.code));
  });
  src.value = "deu_Latn";
  tgt.value = "hrv_Latn";
}

async function checkStatus() {
  const res = await fetch('/status');
  const data = await res.json();
  currentModel = data.model;
  installedModels = data.models;
  fillModels();
  updateSwitchButton();
  const name = models.find(m => m.id === currentModel);
  const badge = document.getElementById('modelBadge');
  const installSec = document.getElementById('installSection');
  if (data.installed) {
    badge.textContent = name.label.split(" - ")[0] + " installed (Ready Offline)";
    badge.className = "badge installed";
    installSec.style.display = "none";
  } else {
    badge.textContent = name.label.split(" - ")[0] + " not installed (" + name.size + ")";
    badge.className = "badge not-installed";
    installSec.style.display = "block";
  }
}

async function installModel() {
  const status = document.getElementById('status');
  setWorking(true);
  status.textContent = "Downloading the selected model... Check the black terminal window for download bars.";
  try {
    const res = await fetch('/install', { method: 'POST' });
    const data = await res.json();
    status.textContent = data.message;
  } catch (e) {
    status.textContent = "Error communicating with local server.";
  }
  setWorking(false);
  checkStatus();
}

async function translatePDF() {
  const fileInput = document.getElementById('pdfFile');
  const status = document.getElementById('status');
  const dlSec = document.getElementById('downloadSection');

  if (!fileInput.files.length) {
    alert("Please select a PDF file first.");
    return;
  }
  if (document.getElementById('srcLang').value === document.getElementById('tgtLang').value) {
    alert("Source and target language must be different.");
    return;
  }

  setWorking(true);
  dlSec.style.display = "none";
  status.textContent = "Running layout detection & translating... Watch the black terminal window for live progress!";

  const formData = new FormData();
  formData.append("file", fileInput.files.item(0));
  formData.append("src_lang", document.getElementById('srcLang').value);
  formData.append("tgt_lang", document.getElementById('tgtLang').value);
  formData.append("ocr", document.getElementById('ocrBox').checked ? "1" : "0");

  try {
    const res = await fetch('/translate_pdf', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok || data.error) {
      status.textContent = "Error: " + (data.error || "Translation failed.");
    } else {
      jobId = data.job_id;
      status.textContent = "Translation complete! " + (data.note ? data.note + " " : "") + "Choose which formatted PDF to download below:";
      dlSec.style.display = "grid";
      checkStatus();
    }
  } catch (e) {
    status.textContent = "Error communicating with local server.";
  }
  setWorking(false);
}

function downloadResult(mode) {
  if (!jobId) return;
  window.location.href = `/download/${jobId}/${mode}`;
}

populateDropdowns();
checkStatus();
</script>
</body>
</html>
"""

JOBS = {}


@app.route("/")
def index():
    return render_template_string(HTML_UI, languages=LANGUAGES, models=MODELS)


@app.route("/status")
def status():
    return jsonify({
        "installed": is_model_cached(),
        "model": MODEL_NAME,
        "models": {name: is_model_cached(name) for name in MODEL_IDS},
    })


@app.route("/model", methods=["POST"])
def change_model():
    name = request.form.get("model", "")
    if name not in MODEL_IDS:
        return jsonify({"message": "Unknown model."}), 400
    if not busy_lock.acquire(blocking=False):
        return jsonify({"message": "A translation is running - switch the model after it has finished."}), 409
    try:
        switch_model(name)
        return jsonify({"message": f"Now using {name}."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"message": f"Could not load {name}: {e} - still using {MODEL_NAME}."}), 500
    finally:
        busy_lock.release()


@app.route("/install", methods=["POST"])
def install():
    try:
        load_engine("deu_Latn")
        return jsonify({"message": "Model installed! You can now translate offline."})
    except Exception as e:
        return jsonify({"message": f"Installation failed: {str(e)}"}), 500


def pdf2zh_command():
    """pdf2zh liegt im Scripts-Ordner der venv, der nicht automatisch im PATH ist."""
    scripts_dir = os.path.dirname(sys.executable)
    for name in ("pdf2zh.exe", "pdf2zh"):
        candidate = os.path.join(scripts_dir, name)
        if os.path.exists(candidate):
            return [candidate]
    found = shutil.which("pdf2zh")
    return [found] if found else ["pdf2zh"]


@app.route("/translate_pdf", methods=["POST"])
def translate_pdf_route():
    if not busy_lock.acquire(blocking=False):
        return jsonify({"error": "The app is busy (translating or switching the model). Please wait."}), 409
    try:
        return translate_pdf()
    finally:
        busy_lock.release()


def translate_pdf():
    global current_src_lang, current_tgt_lang
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    pdf_file = request.files["file"]
    src_lang = request.form.get("src_lang", "deu_Latn")
    tgt_lang = request.form.get("tgt_lang", "hrv_Latn")
    use_ocr = request.form.get("ocr") == "1"
    if src_lang == tgt_lang:
        return jsonify({"error": "Source and target language must be different."}), 400

    with engine_lock:
        current_src_lang = src_lang
        current_tgt_lang = tgt_lang

    src_iso = ISO_MAP.get(src_lang, "de")
    tgt_iso = ISO_MAP.get(tgt_lang, "hr")
    base_name = os.path.splitext(os.path.basename(pdf_file.filename or "document.pdf"))[0] or "document"

    try:
        load_engine(src_lang)
        kept_original.clear()
        work_dir = tempfile.mkdtemp(prefix="pdf2zh_")
        # WICHTIG: pdf2zh LÖSCHT seine Eingabedatei, wenn sie im Temp-Ordner liegt (high_level.py).
        # Deshalb bekommt pdf2zh nur eine Kopie (input.pdf); original.pdf bleibt für OCR & Co. erhalten.
        original_path = os.path.join(work_dir, "original.pdf")
        input_path = os.path.join(work_dir, "input.pdf")
        pdf_file.save(original_path)
        shutil.copyfile(original_path, input_path)

        # Inhaltsverzeichnisse selbst übersetzen (pdf2zh bekommt diese Zeilen nicht zu sehen)
        notes = []
        toc_entries = {}
        base_path = original_path  # Grundlage, falls pdf2zh nichts erzeugt
        try:
            import toc
            stripped_path = os.path.join(work_dir, "stripped.pdf")
            toc_entries = toc.strip_toc(original_path, stripped_path)
            if toc_entries:
                shutil.copyfile(stripped_path, input_path)
                base_path = stripped_path
        except Exception as e:
            traceback.print_exc()
            toc_entries = {}
            shutil.copyfile(original_path, input_path)
            notes.append(f"Table of contents handling skipped: {type(e).__name__}: {e}")

        env = os.environ.copy()
        # pdf2zh spricht über die OpenAI-Schnittstelle mit der Bridge oben (/v1/chat/completions)
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
        env["OPENAI_API_KEY"] = "offline-local"
        env["OPENAI_MODEL"] = "nllb-200"
        # lokale Aufrufe nie über einen Firmen-Proxy leiten
        env["NO_PROXY"] = "127.0.0.1,localhost"
        env["no_proxy"] = "127.0.0.1,localhost"
        env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        cmd = pdf2zh_command() + [
            input_path,
            "--lang-in", src_iso,
            "--lang-out", tgt_iso,
            "--service", "openai:nllb-200",
            "--thread", "1",
            "--vfont", FORMULA_FONT_REGEX,
            "--ignore-cache",  # sonst liefert pdf2zh alte, fehlerhafte Übersetzungen erneut
        ]
        mono_path = os.path.join(work_dir, "input-mono.pdf")
        dual_path = os.path.join(work_dir, "input-dual.pdf")
        base_is_copy = False
        result = None
        if USE_PDF2ZH:
            print("[+] Starting pdf2zh layout engine...")
            result = subprocess.run(cmd, env=env, cwd=work_dir)
        else:
            import layout
            st = layout.translate_layout(base_path, mono_path, tgt_lang, translate_chunk)
            base_is_copy = st["blocks"] == 0
            if st["rotated"] or st["failed"]:
                notes.append(f"Pages left untranslated: {st['rotated']} rotated, {st['failed']} failed.")

        if use_ocr and not os.path.exists(mono_path):
            # z. B. reine Scans: pdf2zh findet keinen Text -> mit der Original-PDF als Basis weitermachen
            shutil.copyfile(base_path, mono_path)
            base_is_copy = True

        if not os.path.exists(mono_path):
            return jsonify({
                "error": f"pdf2zh did not produce an output PDF (exit code {result.returncode}). "
                         "Check the black terminal window for details."
            }), 500

        extra_blocks = 0
        final_mono = mono_path
        if toc_entries:
            try:
                toc_out = os.path.join(work_dir, "input-mono-toc.pdf")
                n_toc = toc.draw_toc(mono_path, toc_out, toc_entries, tgt_lang, translate_chunk)
                final_mono = toc_out
                extra_blocks += n_toc
                notes.append(f"Table of contents: {n_toc} entries translated.")
            except Exception as e:  # ohne diesen Schritt würden die Zeilen fehlen -> Fehler melden
                traceback.print_exc()
                return jsonify({"error": f"Table of contents could not be written: {type(e).__name__}: {e}"}), 500
        if use_ocr:
            try:
                import ocr_images
            except Exception as e:
                ocr_images = None
                notes.append(f"Extra steps skipped: {type(e).__name__}: {e}")

            if ocr_images is not None and USE_PDF2ZH:
                # Schritt A: echter Text, den pdf2zh nicht angefasst hat (Grafiken, Kopf-/Fußzeilen, Überschriften).
                # Braucht kein OCR - der Text wird direkt aus der PDF gelesen.
                try:
                    fig_out = os.path.join(work_dir, "input-mono-text.pdf")
                    st = ocr_images.overlay_untranslated_text(
                        original_path, final_mono, fig_out, src_lang, tgt_lang, translate_chunk
                    )
                    if st["blocks"] > 0:
                        final_mono = fig_out
                        extra_blocks += st["blocks"]
                        notes.append(f"Figures/headers/footers: {st['blocks']} text line(s) translated in place.")
                except Exception as e:
                    traceback.print_exc()
                    notes.append(f"Figure text skipped: {type(e).__name__}: {e}")

            if ocr_images is not None:
                # Schritt B: Text in Rasterbildern (Screenshots, Scans) per OCR
                try:
                    ocr_out = os.path.join(work_dir, "input-mono-ocr.pdf")
                    stats = ocr_images.overlay_ocr_translations(
                        original_path, final_mono, ocr_out, src_lang, tgt_lang, translate_chunk
                    )
                    if stats["blocks"] > 0:
                        final_mono = ocr_out
                        extra_blocks += stats["blocks"]
                        notes.append(f"OCR: translated {stats['blocks']} text block(s) in {stats['images']} image(s).")
                    else:
                        notes.append("OCR: no translatable text found in images.")
                except ModuleNotFoundError as e:
                    notes.append(f"OCR skipped: package '{e.name}' is missing (run start_translator.bat again).")
                except Exception as e:
                    traceback.print_exc()
                    notes.append(f"OCR skipped: {type(e).__name__}: {e}")

        if kept_original:
            notes.append(f"{len(kept_original)} passage(s) kept in the original language (model output was unusable).")
        note = " ".join(notes)
        if base_is_copy and extra_blocks == 0:
            return jsonify({
                "error": "Nothing could be translated: no text was found and the extra steps found none. " + note
            }), 500

        if final_mono != mono_path:  # Ergebnis hat sich geändert -> zweisprachige PDF neu bauen
            mono_path = final_mono
            try:
                import ocr_images
                new_dual = os.path.join(work_dir, "input-dual-final.pdf")
                ocr_images.build_side_by_side(original_path, mono_path, new_dual)
                dual_path = new_dual
            except Exception as e:
                print(f"[+] Bilingual PDF could not be rebuilt: {e}")

        if not os.path.exists(dual_path):  # z. B. wenn pdf2zh nur eine Datei erzeugt hat
            try:
                import ocr_images
                ocr_images.build_side_by_side(original_path, mono_path, dual_path)
            except Exception as e:
                print(f"[+] Bilingual PDF could not be created: {e}")

        job_id = os.path.basename(work_dir)
        JOBS[job_id] = {"mono": mono_path, "dual": dual_path, "name": base_name}
        return jsonify({"job_id": job_id, "note": note})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download/<job_id>/<mode>")
def download(job_id, mode):
    job = JOBS.get(job_id)
    if not job or mode not in ("mono", "dual") or not os.path.exists(job[mode]):
        return "File not found", 404
    suffix = "translated" if mode == "mono" else "bilingual"
    return send_file(
        job[mode],
        as_attachment=True,
        mimetype="application/pdf",
        download_name=f"{job['name']}_{suffix}.pdf",
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)