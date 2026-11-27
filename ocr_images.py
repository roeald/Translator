"""
Zusatz-Schritte für den Offline-PDF-Übersetzer (alles, was pdf2zh nicht übersetzt).

1) overlay_untranslated_text(): ECHTER Text, den pdf2zh unangetastet lässt (Text in
   Abbildungen/Vektorgrafiken, Kopf-/Fußzeilen, manche Überschriften). Er wird direkt aus
   der PDF gelesen (kein OCR nötig), übersetzt, das Original entfernt und die Übersetzung
   an derselben Stelle (gleiche Größe, Farbe, Ausrichtung) eingesetzt.

2) overlay_ocr_translations(): Text, der als BILD (Screenshot, Scan, Foto) in der PDF
   steckt. Er wird mit EasyOCR erkannt, mit derselben Übersetzungsfunktion übersetzt und
   auf die übersetzte PDF gelegt: Der Originaltext wird mit der Hintergrundfarbe abgedeckt
   und die Übersetzung in der erkannten Textfarbe darübergeschrieben.

Wichtig: Fehler in diesem Modul dürfen die normale Übersetzung nie zerstören.
Jedes Bild wird deshalb einzeln abgesichert, und bei Problemen bleibt das Original-Bild.
"""
import difflib
import glob
import os
import re
import threading

import numpy as np

try:
    import pymupdf  # PyMuPDF >= 1.24.3
except ImportError:  # ältere PyMuPDF-Versionen
    import fitz as pymupdf

# NLLB-Sprachcode -> EasyOCR-Sprachen (nicht alle Schriften lassen sich mischen)
OCR_LANGS = {
    "deu_Latn": ["de"],
    "hrv_Latn": ["hr"],
    "eng_Latn": ["en"],
    "nld_Latn": ["nl"],
    "pol_Latn": ["pl"],
    "rus_Cyrl": ["ru", "en"],
    "tur_Latn": ["tr"],
    "jpn_Jpan": ["ja", "en"],
    "zho_Hans": ["ch_sim", "en"],
}
NO_SPACE_LANGS = {"jpn_Jpan", "zho_Hans"}

MIN_CONF = 0.35          # unsichere OCR-Treffer ignorieren
MIN_W_PT = 40            # kleinere Bilder (Icons, Linien) ignorieren
MIN_H_PT = 20
MIN_PX = 50
TARGET_PX = 1800         # Ziel-Auflösung (längste Seite) für die Texterkennung
MIN_FONT_SIZE = 4.0
HYPHEN_KEEP_WORDS = {"und", "oder", "bzw", "bzw.", "sowie", "bis", "and", "or"}

_readers = {}
_reader_lock = threading.Lock()


# --------------------------------------------------------------------------- OCR

def get_reader(src_lang):
    """EasyOCR-Reader pro Sprache nur einmal laden (erste Nutzung lädt ~100 MB Modelle)."""
    langs = tuple(OCR_LANGS.get(src_lang, ["en"]))
    with _reader_lock:
        if langs not in _readers:
            import easyocr  # erst hier importieren: spart Startzeit, Fehler werden gemeldet
            print(f"[OCR] Loading EasyOCR models for {list(langs)} (first use downloads them)...")
            _readers[langs] = easyocr.Reader(list(langs), gpu=False)
        return _readers[langs]


def _render_region(page, rect):
    """Rendert den Bildbereich der Seite als RGB-Array. Gibt (array, breite, höhe) zurück."""
    longest = max(rect.width, rect.height)
    dpi = int(max(150, min(300, TARGET_PX * 72 / longest)))
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif pix.n >= 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr), pix.width, pix.height


def _ocr_lines(reader, arr):
    """Führt OCR aus und liefert Zeilen als Dicts mit Pixel-Koordinaten."""
    lines = []
    for box, text, conf in reader.readtext(arr):
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        dx = abs(float(box[1][0]) - float(box[0][0]))
        dy = abs(float(box[1][1]) - float(box[0][1]))
        if dx == 0 or dy / dx > 0.25:  # gedrehter Text: nicht sauber überlagerbar
            continue
        text = " ".join(str(text).split())
        if float(conf) < MIN_CONF or not any(c.isalpha() for c in text):
            continue
        lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    return lines


# ------------------------------------------------------- Zeilen zu Absätzen bündeln

def _can_follow(a, b):
    """Gehört Zeile b als Fortsetzung zu Zeile a (gleicher Absatz)?"""
    ha, hb = a["y1"] - a["y0"], b["y1"] - b["y0"]
    if ha <= 0 or hb <= 0 or b["y0"] <= a["y0"]:
        return False
    gap = b["y0"] - a["y1"]
    if gap < -0.45 * ha or gap > 0.7 * max(ha, hb):  # OCR-Boxen eines Absatzes überlappen oft leicht
        return False
    if not 0.6 <= hb / ha <= 1.6:
        return False
    left = abs(b["x0"] - a["x0"]) <= 1.5 * ha
    right = abs(b["x1"] - a["x1"]) <= 0.5 * ha  # rechtsbündige Diagramm-Beschriftung
    centered = abs((a["x0"] + a["x1"]) - (b["x0"] + b["x1"])) <= 1.6 * ha
    if not (left or right or centered):
        return False
    end = a["text"].rstrip()[-1:]
    if end == ":":
        return False  # "Das Deckblatt enthält:" / "Name:": danach folgt etwas Eigenständiges
    if end in ".;!?" and b["text"][:1].isupper():
        return False  # Satzende + Großbuchstabe: neuer Absatz
    return True


def _join_lines(texts):
    out = texts[0]
    for t in texts[1:]:
        first = t.split()[0].lower() if t.split() else ""
        if out.endswith("-") and t[:1].islower() and first not in HYPHEN_KEEP_WORDS:
            out = out[:-1] + t  # Silbentrennung am Zeilenende auflösen
        else:
            out += " " + t
    return out


def merge_rows(lines):
    """Teile derselben Zeile verbinden (EasyOCR trennt z. B. an "-" oder "/")."""
    out = []
    for line in sorted(lines, key=lambda l: l["x0"]):
        for o in out:
            ho, hl = o["y1"] - o["y0"], line["y1"] - line["y0"]
            overlap = min(o["y1"], line["y1"]) - max(o["y0"], line["y0"])
            h = min(ho, hl)
            if overlap >= 0.6 * h and -0.3 * h <= line["x0"] - o["x1"] <= 0.9 * h and max(ho, hl) <= 1.6 * h:
                o["text"] += " " + line["text"]
                o["x1"] = max(o["x1"], line["x1"])
                o["y0"], o["y1"] = min(o["y0"], line["y0"]), max(o["y1"], line["y1"])
                break
        else:
            out.append(dict(line))
    return out


def _tightest_neighbour(line, lines):
    """Kleinster Abstand zu einer passenden Zeile direkt darüber oder darunter."""
    gaps = [o["y0"] - line["y1"] for o in lines if o is not line and _can_follow(line, o)]
    gaps += [line["y0"] - o["y1"] for o in lines if o is not line and _can_follow(o, line)]
    return min(gaps, default=None)


def group_lines(lines):
    """Fasst eng untereinanderstehende Zeilen zu Blöcken zusammen."""
    lines = merge_rows(lines)
    lines = sorted(lines, key=lambda l: (l["y0"], l["x0"]))
    used = [False] * len(lines)
    blocks = []
    for i in range(len(lines)):
        if used[i]:
            continue
        used[i] = True
        members = [lines[i]]
        while True:
            last = members[-1]
            best_j, best_gap = None, None
            for j in range(len(lines)):
                if used[j] or not _can_follow(last, lines[j]):
                    continue
                gap = lines[j]["y0"] - last["y1"]
                # Hat eine der beiden Zeilen eine deutlich engere Nachbarzeile, sind es zwei Beschriftungen
                limit = gap - 0.3 * (last["y1"] - last["y0"])
                if any(t is not None and t < limit for t in
                       (_tightest_neighbour(last, lines), _tightest_neighbour(lines[j], lines))):
                    continue
                if best_j is None or gap < best_gap:
                    best_j, best_gap = j, gap
            if best_j is None:
                break
            used[best_j] = True
            members.append(lines[best_j])
        blocks.append({
            "lines": members,
            "x0": min(l["x0"] for l in members),
            "y0": min(l["y0"] for l in members),
            "x1": max(l["x1"] for l in members),
            "y1": max(l["y1"] for l in members),
            "text": _join_lines([l["text"] for l in members]),
        })
    return blocks


# ------------------------------------------------------------- Farben / Layout

def estimate_colors(arr, x0, y0, x1, y1):
    """Schätzt Hintergrund- und Textfarbe (RGB 0-255) in einem Pixelbereich."""
    h, w = arr.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return (255, 255, 255), (0, 0, 0)
    region = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    q = region // 32
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    bg = region[keys == np.bincount(keys, minlength=512).argmax()].mean(axis=0)
    dist = np.abs(region - bg).sum(axis=1)
    dmax = dist.max()
    lum = lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    fg = region[dist >= 0.7 * dmax].mean(axis=0) if dmax >= 120 else None
    if fg is None or abs(lum(fg) - lum(bg)) < 90:  # zu wenig Kontrast: Schwarz oder Weiß
        fg = np.array([0, 0, 0]) if lum(bg) > 128 else np.array([255, 255, 255])
    return tuple(int(v) for v in bg), tuple(int(v) for v in fg)


def _choose_align(block, img_w):
    """0 = linksbündig, 1 = zentriert (für zentrierte Titel im Bild)."""
    lines = block["lines"]
    if len(lines) == 1:
        cx = (block["x0"] + block["x1"]) / 2
        return 1 if abs(cx - img_w / 2) < 0.04 * img_w else 0
    centers = [(l["x0"] + l["x1"]) / 2 for l in lines]
    lefts = [l["x0"] for l in lines]
    hh = sum(l["y1"] - l["y0"] for l in lines) / len(lines)
    if max(centers) - min(centers) < 0.5 * hh and max(lefts) - min(lefts) > 0.5 * hh:
        return 1
    return 0


def _text_box_px(b, others, img_w, align, lo=0):
    """Einzeilige Texte dürfen nach rechts (bzw. zu beiden Seiten) etwas breiter werden.
    align: 0 = links, 1 = zentriert, 2 = rechts. lo = linke Grenze (Standard 0)."""
    x0, x1 = b["x0"], b["x1"]
    if len(b["lines"]) > 1:
        return (x0, b["y0"], x1, b["y1"])
    extra = 0.35 * (x1 - x0)
    right_limit, left_limit = img_w, lo
    for o in others:
        if o is b or o["y1"] <= b["y0"] or o["y0"] >= b["y1"]:
            continue
        if o["x0"] >= b["x1"]:
            right_limit = min(right_limit, o["x0"] - 4)
        elif o["x1"] <= b["x0"]:
            left_limit = max(left_limit, o["x1"] + 4)
    if align == 2:  # rechtsbündig: nach links wachsen
        return (min(max(left_limit, x0 - extra), x0), b["y0"], x1, b["y1"])
    if align == 1:
        left = max(left_limit, x0 - extra / 2)
        right = min(right_limit, x1 + extra / 2)
        return (min(left, x0), b["y0"], max(right, x1), b["y1"])
    return (x0, b["y0"], max(min(right_limit, x1 + extra), x1), b["y1"])


def _ocr_align(block, blocks, img_w):
    """Wie _choose_align, erkennt zusätzlich rechtsbündige Beschriftungen (2)."""
    lines = block["lines"]
    h = sum(l["y1"] - l["y0"] for l in lines) / len(lines)
    peers = lines if len(lines) > 1 else [
        o for o in blocks if abs(o["x1"] - block["x1"]) <= 0.25 * h and abs(o["y0"] - block["y0"]) < 8 * h]
    if len(peers) > 1:
        rights = [l["x1"] for l in peers]
        lefts = [l["x0"] for l in peers]
        if max(rights) - min(rights) <= 0.4 * h and max(lefts) - min(lefts) > 0.8 * h:
            return 2
    return _choose_align(block, img_w)


def _translatable(text):
    """Achsenkürzel wie "R1", "Rl" oder "1o" nie übersetzen (sonst erfindet das Modell Wörter)."""
    return re.search(r"[^\W\d_]{3,}", text) is not None


def _barely_changed(original, translated):
    a, b = _norm(original), _norm(translated)
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() > 0.8


# ------------------------------------------------------------------- PDF-Zeichnen

def _find_font(tgt_lang, bold=False):
    win = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    babel = os.path.join(os.path.expanduser("~"), ".cache", "babeldoc", "fonts")
    if tgt_lang == "zho_Hans":
        cands = [os.path.join(win, n) for n in (("msyhbd.ttc",) if bold else ()) + ("msyh.ttc", "simsun.ttc", "simhei.ttf")]
        cands += glob.glob(os.path.join(babel, "SourceHan*")) + glob.glob(os.path.join(babel, "*CN*"))
    elif tgt_lang == "jpn_Jpan":
        cands = [os.path.join(win, n) for n in (("YuGothB.ttc",) if bold else ()) + ("YuGothR.ttc", "meiryo.ttc", "msgothic.ttc")]
        cands += glob.glob(os.path.join(babel, "SourceHan*")) + glob.glob(os.path.join(babel, "*JP*"))
    else:  # Latein/Kyrillisch: Arial hat č ć đ š ž ł ğ ş und kyrillische Buchstaben
        names = ("arialbd.ttf", "segoeuib.ttf", "tahomabd.ttf") if bold else ()
        cands = [os.path.join(win, n) for n in names + ("arial.ttf", "segoeui.ttf", "tahoma.ttf")]
        cands += [
            os.path.join(babel, "GoNotoKurrent-Regular.ttf"),
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


_fonts = {}


def _load_font(tgt_lang, bold=False):
    key = (tgt_lang, bold)
    if key in _fonts:
        return _fonts[key]
    path = _find_font(tgt_lang, bold)
    font = None
    if path:
        try:
            font = pymupdf.Font(fontfile=path)
        except Exception as e:
            print(f"[OCR] Font {path} could not be loaded: {e}")
    if font is None:
        print("[OCR] No suitable font found - special characters may be missing in the translated text.")
        font = pymupdf.Font("helv")
    _fonts[key] = font
    return font


def _px_to_rect(img_rect, pw, ph, x0, y0, x1, y1):
    fx, fy = img_rect.width / pw, img_rect.height / ph
    return pymupdf.Rect(img_rect.x0 + x0 * fx, img_rect.y0 + y0 * fy,
                        img_rect.x0 + x1 * fx, img_rect.y0 + y1 * fy)


def _scale(rect, sx, sy):
    return pymupdf.Rect(rect.x0 * sx, rect.y0 * sy, rect.x1 * sx, rect.y1 * sy)


def _fit_text(page, font, rect, text, align, n_lines, start=None, min_fs=MIN_FONT_SIZE):
    """Verkleinert die Schrift, bis der Text in die Box passt. Gibt TextWriter zurück oder None."""
    fs = float(start) if start else min(60.0, rect.height / max(n_lines, 1) * 0.9)
    words = text.split() if " " in text else []
    while fs >= min_fs:
        if words and max(font.text_length(w, fontsize=fs) for w in words) > rect.width:
            fs *= 0.92  # lieber kleiner als ein Wort mitten durchtrennen
            continue
        tw = pymupdf.TextWriter(page.rect)
        try:
            overflow = tw.fill_textbox(rect, text, font=font, fontsize=fs, align=align)
        except Exception:
            overflow = True
        if not overflow:
            return tw
        fs *= 0.92
    return None


def _covered_by_real_text(rect, words):
    """Liegt über dem Bildbereich echter PDF-Text? Dann hat pdf2zh ihn schon übersetzt."""
    area = rect.get_area()
    if area <= 0:
        return False
    covered = 0.0
    for w in words:
        inter = rect & w
        if not inter.is_empty:
            covered += inter.get_area()
    return covered > 0.25 * area


def _norm(s):
    return re.sub(r"\W+", "", s).casefold()


def _prepare_block(mpage, font, arr, b, others, text, align, text_px, img_rect, pw, ph, sx, sy):
    """Berechnet Abdeckfläche und Text eines Blocks. None = passt nicht, Original behalten."""
    img_m = _scale(img_rect, sx, sy)
    cover = _scale(_px_to_rect(img_rect, pw, ph, b["x0"] - 3, b["y0"] - 2, b["x1"] + 3, b["y1"] + 2), sx, sy) & img_m
    trect = _scale(_px_to_rect(img_rect, pw, ph, *text_px), sx, sy) & img_m
    if cover.is_empty or trect.is_empty:
        return None
    n_lines = len(b["lines"])
    line_h = sum(l["y1"] - l["y0"] for l in b["lines"]) / n_lines * img_rect.height / ph * sy
    start = min(60.0, 0.82 * line_h)  # ungefähr die Originalschriftgröße
    # etwas Luft nach unten für Unterlängen, aber nie in den nächsten Textblock hinein
    below = [o["y0"] for o in others if o is not b and o["y0"] >= b["y1"] - 2
             and o["x0"] < text_px[2] and o["x1"] > text_px[0]]
    limit = _px_to_rect(img_rect, pw, ph, 0, min(below, default=ph), 0, ph).y0 * sy
    roomy = pymupdf.Rect(trect.x0, trect.y0, trect.x1,
                         max(trect.y1, min(trect.y1 + 0.5 * line_h, limit))) & img_m
    tw = _fit_text(mpage, font, roomy, text, align, n_lines, start=start)
    if tw is None:
        return None  # passt nirgends: Original im Bild lassen
    bg, fg = estimate_colors(arr, b["x0"] - 2, b["y0"] - 2, b["x1"] + 2, b["y1"] + 2)
    return cover, tuple(c / 255 for c in bg), tw, tuple(c / 255 for c in fg)


# ---------------------------------------------------------------------- Hauptfunktion

def _collect_images(doc, n_pages):
    items, counts = [], {}
    for pno in range(n_pages):
        page = doc[pno]
        if page.rotation != 0:
            continue
        try:
            infos = page.get_image_info(xrefs=True)
        except Exception:
            continue
        for info in infos:
            try:
                rect = pymupdf.Rect(info["bbox"]) & page.rect
            except Exception:
                continue
            if rect.is_empty or rect.width < MIN_W_PT or rect.height < MIN_H_PT:
                continue
            if min(info.get("width", 0), info.get("height", 0)) < MIN_PX:
                continue
            xref = info.get("xref", 0) or 0
            items.append((pno, rect, xref))
            if xref:
                counts[xref] = counts.get(xref, 0) + 1
    return items, counts


def overlay_ocr_translations(src_path, mono_path, out_path, src_lang, tgt_lang, translate_fn, log=print):
    """
    Liest die Bilder der Original-PDF (src_path), übersetzt erkannten Text mit translate_fn
    und legt ihn auf die übersetzte PDF (mono_path). Ergebnis nach out_path, aber nur wenn
    mindestens ein Textblock ersetzt wurde. Gibt Statistik-Dict zurück.
    """
    stats = {"images": 0, "blocks": 0, "skipped": 0, "errors": 0}
    src = pymupdf.open(src_path)
    mono = pymupdf.open(mono_path)
    try:
        n = min(len(src), len(mono))
        items, counts = _collect_images(src, n)
        # Bilder, die auf vielen Seiten identisch vorkommen (Logo, Briefkopf): nicht übersetzen
        items = [it for it in items if not (it[2] and counts.get(it[2], 0) > 3 and n > 3)]
        if not items:
            log("[OCR] No images found in the PDF.")
            return stats

        reader = get_reader(src_lang)
        font = _load_font(tgt_lang)
        words_cache = {}

        for idx, (pno, img_rect, xref) in enumerate(items, 1):
            try:
                page, mpage = src[pno], mono[pno]
                sx = mpage.rect.width / page.rect.width
                sy = mpage.rect.height / page.rect.height
                log(f"[OCR] Image {idx}/{len(items)} on page {pno + 1}: recognizing text...")
                arr, pw, ph = _render_region(page, img_rect)
                lines = _ocr_lines(reader, arr)
                if not lines:
                    continue
                if pno not in words_cache:
                    words_cache[pno] = [pymupdf.Rect(w[:4]) for w in page.get_text("words")]

                todo = []
                blocks = group_lines(lines)
                for b in blocks:
                    if not _translatable(b["text"]):
                        stats["skipped"] += 1
                        continue
                    brect = _px_to_rect(img_rect, pw, ph, b["x0"], b["y0"], b["x1"], b["y1"])
                    if _covered_by_real_text(brect, words_cache[pno]):
                        stats["skipped"] += 1
                        continue
                    translated = " ".join(str(translate_fn(b["text"])).split())
                    if (not translated or _barely_changed(b["text"], translated)
                            or len(translated) > 3 * len(b["text"]) + 30):
                        stats["skipped"] += 1  # unverändert: das scharfe Original behalten
                        continue
                    todo.append((b, translated))

                prepared = []
                for b, translated in todo:
                    align = _ocr_align(b, blocks, pw)
                    # Nachbarn sind ALLE erkannten Texte, auch die nicht übersetzten
                    x0, y0, x1, y1 = _text_box_px(b, blocks, pw, align)
                    if len(b["lines"]) > 1 and align == 2:
                        x0 = max(0, x0 - 0.15 * (x1 - x0))
                    item = _prepare_block(mpage, font, arr, b, blocks, translated, align, (x0, y0, x1, y1),
                                          img_rect, pw, ph, sx, sy)
                    if item:
                        prepared.append(item)
                    else:
                        stats["skipped"] += 1
                # erst alle Flächen abdecken, dann schreiben - so übermalt kein Block den Nachbartext
                for cover, bg, _tw, _fg in prepared:
                    mpage.draw_rect(cover, color=None, fill=bg, width=0, overlay=True)
                for _cover, _bg, tw, fg in prepared:
                    tw.write_text(mpage, color=fg, overlay=True)
                drawn = len(prepared)
                if drawn:
                    stats["images"] += 1
                    stats["blocks"] += drawn
            except Exception as e:  # ein defektes Bild darf die restliche Arbeit nicht stoppen
                stats["errors"] += 1
                log(f"[OCR] Image {idx} skipped: {type(e).__name__}: {e}")

        if stats["blocks"] > 0:
            try:
                mono.subset_fonts()  # Schriftarten verkleinern (optional)
            except Exception:
                pass
            mono.save(out_path, garbage=3, deflate=True)
        return stats
    finally:
        src.close()
        mono.close()


# ------------------------------------------------ Echter Text, den pdf2zh nicht übersetzt
# pdf2zh lässt bewusst vieles unangetastet: Text in Abbildungen und Vektorgrafiken (z. B. ein
# nachgebautes Deckblatt), Kopf-/Fußzeilen und manche Überschriften. Das ist ECHTER Text
# (kein Bild) und braucht kein OCR: Wir erkennen ihn daran, dass er in der übersetzten PDF
# noch genauso an derselben Stelle steht wie im Original.

FORMULA_CHARS = re.compile(r"[=∑∫√±×÷≤≥≠≈∞∂∇∈∀∃→←↔]")
FORMULA_FONTS = ("cmmi", "cmsy", "cmex", "math", "symbol", "stix", "mtmi", "mtsy", "mtex")
MIN_REL_FONT = 0.55  # Übersetzung darf höchstens auf 55 % der Originalschrift schrumpfen


def _rgb_from_int(c):
    return (((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255)


def _page_segments(page):
    """Textzeilen einer Seite als Segmente. Weit auseinanderstehende Teile einer Zeile
    (z. B. 'Stand: April 2026 ........ Seite 5 von 17') werden getrennt."""
    segs = []
    try:
        d = page.get_text("dict")
    except Exception:
        return segs
    for blk in d.get("blocks", []):
        if blk.get("type") != 0:
            continue
        for line in blk.get("lines", []):
            dx, dy = line.get("dir", (1, 0))
            if dx <= 0 or abs(dy) > 0.05:  # nur waagerechter Text von links nach rechts
                continue
            cur = None
            for sp in line.get("spans", []):
                t = sp.get("text", "")
                if not t.strip():
                    if cur is not None:
                        cur["text"] += t
                    continue
                x0, y0, x1, y1 = sp["bbox"]
                size = float(sp.get("size", 10) or 10)
                if cur is not None and x0 - cur["x1"] <= max(1.5 * cur["size"], 8.0):
                    cur["text"] += t
                    cur["x1"] = max(cur["x1"], x1)
                    cur["y0"], cur["y1"] = min(cur["y0"], y0), max(cur["y1"], y1)
                    continue
                if cur is not None:
                    segs.append(cur)
                font = str(sp.get("font", "") or "")
                base = float(sp["origin"][1]) if sp.get("origin") else y1 - 0.2 * size
                cur = {"text": t, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "size": size, "base": base,
                       "color": int(sp.get("color", 0) or 0), "font": font,
                       "bold": bool(int(sp.get("flags", 0)) & 16) or "bold" in font.lower()}
            if cur is not None:
                segs.append(cur)
    for s in segs:
        s["text"] = " ".join(s["text"].split())
    return [s for s in segs if s["text"] and s["x1"] > s["x0"] and s["y1"] > s["y0"]]


def _untouched_segments(mono_segs, src_segs, sx, sy):
    """Segmente der übersetzten Seite, die genauso (Text + Position) im Original stehen."""
    by_text = {}
    for s in src_segs:
        by_text.setdefault(_norm(s["text"]), []).append(s)
    out = []
    for m in mono_segs:
        for s in by_text.get(_norm(m["text"]), []):
            if abs(s["x0"] * sx - m["x0"]) <= 3 and abs(s["y0"] * sy - m["y0"]) <= 3:
                out.append(m)
                break
    return out


def _worth_translating(seg):
    t = seg["text"]
    if sum(c.isalpha() for c in t) < 2:
        return False  # Zahlen, Satzzeichen, einzelne Buchstaben
    if FORMULA_CHARS.search(t) or any(k in seg["font"].lower() for k in FORMULA_FONTS):
        return False  # Formeln nicht durch das Sprachmodell schicken
    if re.search(r"\.{4,}|…{2,}", t):
        return False  # Inhaltsverzeichnis-Zeilen mit Punktlinie
    if " " not in t and re.search(r"https?://|www\.|\S@\S", t):
        return False  # URL / E-Mail
    return True


def _merge_segments(segs):
    """Fasst Fortsetzungszeilen (Absatz in einem Textfeld) zusammen. Bewusst vorsichtig:
    Nur wenn die nächste Zeile klein beginnt oder die vorige mit Bindestrich endet."""
    segs = sorted(segs, key=lambda s: (s["y0"], s["x0"]))
    blocks = []
    for s in segs:
        target = None
        for b in blocks[::-1][:30]:
            last = b["lines"][-1]
            h = last["y1"] - last["y0"]
            gap = s["y0"] - last["y1"]
            if h <= 0 or not (-0.2 * h <= gap <= 0.6 * h):
                continue
            if abs(s["size"] - last["size"]) > 0.6 or s["bold"] != last["bold"]:
                continue
            same_left = abs(s["x0"] - last["x0"]) <= 2
            same_center = abs((s["x0"] + s["x1"]) / 2 - (last["x0"] + last["x1"]) / 2) <= 2
            if not (same_left or same_center):
                continue
            prev = last["text"].rstrip()
            if prev.endswith((":", ".", ";", "!", "?")):
                continue
            if not (s["text"][:1].islower() or prev.endswith("-")):
                continue
            target = b
            break
        if target is None:
            blocks.append({"lines": [s]})
        else:
            target["lines"].append(s)
    for b in blocks:
        ls = b["lines"]
        b["x0"], b["y0"] = min(l["x0"] for l in ls), min(l["y0"] for l in ls)
        b["x1"], b["y1"] = max(l["x1"] for l in ls), max(l["y1"] for l in ls)
        b["text"] = _join_lines([l["text"] for l in ls])
        b["size"], b["color"], b["bold"] = ls[0]["size"], ls[0]["color"], ls[0]["bold"]
    return blocks


def _block_align_pt(b, blocks, page_w, left_margin):
    """0 = links, 1 = zentriert, 2 = rechtsbündig (aus der Lage der Nachbarzeilen abgeleitet)."""
    if len(b["lines"]) > 1:
        return _choose_align(b, 0)
    cx = (b["x0"] + b["x1"]) / 2
    centered = left_aligned = False
    for o in blocks:
        if o is b or abs(o["y0"] - b["y0"]) > 300:
            continue
        ocx = (o["x0"] + o["x1"]) / 2
        if abs(ocx - cx) <= 1.5 and abs(o["x0"] - b["x0"]) > 3:
            centered = True
        if abs(o["x0"] - b["x0"]) <= 1.5 and abs(ocx - cx) > 3:
            left_aligned = True
    if centered and not left_aligned:
        return 1
    if (not left_aligned and b["x0"] > page_w / 2
            and abs(b["x1"] - (page_w - left_margin)) <= 6):
        return 2  # z. B. "Seite 5 von 17" am rechten Rand
    return 0


def _apply_redactions(page):
    """Entfernt den markierten Text, lässt aber Bilder und Linien/Flächen unangetastet."""
    img_none = getattr(pymupdf, "PDF_REDACT_IMAGE_NONE", 0)
    art_none = getattr(pymupdf, "PDF_REDACT_LINE_ART_NONE", 0)
    for kw in ({"images": img_none, "graphics": art_none}, {"images": img_none}):
        try:
            page.apply_redactions(**kw)
            return True
        except TypeError:  # ältere PyMuPDF-Version kennt den Parameter nicht
            continue
    return False


def _wrap(font, text, fs, width):
    """Einfacher Zeilenumbruch nach Wörtern."""
    lines, cur = [], ""
    for w in text.split():
        t = w if not cur else cur + " " + w
        if not cur or font.text_length(t, fontsize=fs) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _layout_block(font, text, b, align, L, R, min_fs):
    """Setzt die Übersetzung auf die Original-Grundlinien (Größe wird nur bei Bedarf verkleinert).
    L..R ist der erlaubte Bereich in pt. Gibt (Größe, [(x, y_grundlinie, text), ...]) oder None zurück."""
    n, size = len(b["lines"]), b["size"]
    ox0, ox1 = b["x0"], b["x1"]
    cx = (ox0 + ox1) / 2
    if align == 1:
        width = 2 * min(cx - L, R - cx)
    elif align == 2:
        width = ox1 - L
    else:
        width = R - ox0
    width = max(width, ox1 - ox0)
    bases = [l["base"] for l in b["lines"]]
    pitch = (bases[-1] - bases[0]) / (n - 1) if n > 1 else 0.0
    fs = size
    while fs >= min_fs:
        lines = [text] if n == 1 else _wrap(font, text, fs, width)
        if len(lines) <= n and all(font.text_length(t, fontsize=fs) <= width + 0.01 for t in lines):
            placed = []
            for i, t in enumerate(lines):
                tl = font.text_length(t, fontsize=fs)
                x = ox0 if align == 0 else (cx - tl / 2 if align == 1 else ox1 - tl)
                placed.append((x, bases[0] + i * pitch, t))
            return fs, placed
        fs *= 0.95
    return None


def _clip_to_free_space(arr, zx, zy, b, L, R, bg, thr=90):
    """Begrenzt den erlaubten Bereich L..R, damit die (evtl. längere) Übersetzung keine Linien,
    Rahmen oder Nachbartexte überdeckt. Gemessen wird im gerenderten Seitenbild."""
    h, w = arr.shape[:2]
    lh = b["y1"] - b["y0"]
    ya = max(0, int((b["y0"] + 0.2 * lh) * zy))
    yb = min(h, max(int((b["y1"] - 0.2 * lh) * zy), ya + 1))
    bgv = np.array(bg, dtype=np.int32)

    def ink_cols(c0, c1):
        reg = arr[ya:yb, c0:c1].astype(np.int32)
        return (np.abs(reg - bgv).sum(axis=2) > thr).any(axis=0)

    xs, xe = int(np.ceil(b["x1"] * zx)) + 2, min(w, int(R * zx))
    if xe > xs:
        ink = ink_cols(xs, xe)
        if ink.any():
            R = min(R, (xs + int(ink.argmax())) / zx - 2)
    xs, xe = int(b["x0"] * zx) - 2, max(0, int(L * zx))
    if xs > xe:
        ink = ink_cols(xe, xs)
        if ink.any():
            last = len(ink) - 1 - int(ink[::-1].argmax())
            L = max(L, (xe + last + 1) / zx + 2)
    return min(L, b["x0"]), max(R, b["x1"])


def _translate_page_text(spage, mpage, tgt_lang, translate_fn, cache, stats, log):
    sx = mpage.rect.width / spage.rect.width
    sy = mpage.rect.height / spage.rect.height
    mono_segs = _page_segments(mpage)
    cand = [s for s in _untouched_segments(mono_segs, _page_segments(spage), sx, sy)
            if _worth_translating(s)]
    if not cand:
        return 0
    blocks = _merge_segments(cand)
    page_w = mpage.rect.width
    x0s = sorted(s["x0"] for s in mono_segs)
    left_margin = x0s[len(x0s) // 4] if x0s else 0
    edge = 20.0  # so nah darf Text an den Seitenrand wachsen

    # Seitenbild VOR allen Änderungen: Hintergrundfarben und freier Platz neben dem Text
    arr, pw, ph = _render_region(mpage, mpage.rect)
    zx, zy = pw / mpage.rect.width, ph / mpage.rect.height
    try:
        img_rects = [pymupdf.Rect(i["bbox"]) for i in mpage.get_image_info()]
    except Exception:
        img_rects = []

    # 1) übersetzen und prüfen, ob die Übersetzung an die Stelle passt (Seite bleibt unverändert)
    queue = []
    for b in blocks:
        try:
            if b["text"] not in cache:
                cache[b["text"]] = " ".join(str(translate_fn(b["text"])).split())
            translated = cache[b["text"]]
        except Exception as e:
            stats["errors"] += 1
            log(f"[Figures] Translation failed for {b['text'][:40]!r}: {type(e).__name__}: {e}")
            continue
        if (not translated or _norm(translated) == _norm(b["text"])
                or len(translated) > 3 * len(b["text"]) + 30):
            stats["skipped"] += 1  # unverändert (Name, Zahl, Fachwort) oder unplausibel
            continue
        align = _block_align_pt(b, blocks, page_w, left_margin)
        L, _y0, R, _y1 = _text_box_px(b, blocks, page_w - edge, align, edge)
        bg, _fg = estimate_colors(arr, b["x0"] * zx, b["y0"] * zy, b["x1"] * zx, b["y1"] * zy)
        L, R = _clip_to_free_space(arr, zx, zy, b, L, R, bg)
        font = _load_font(tgt_lang, b["bold"])
        lay = _layout_block(font, translated, b, align, L, R, max(MIN_FONT_SIZE, MIN_REL_FONT * b["size"]))
        if lay is None:
            stats["skipped"] += 1  # passt nirgends: lieber das Original stehen lassen
            continue
        queue.append((b, font, lay))
    if not queue:
        return 0

    # 2) Originaltext entfernen (Redaktion); Grafiken, Linien und Bilder bleiben
    line_rects = []
    for b, _f, _l in queue:
        for l in b["lines"]:
            h = l["y1"] - l["y0"]
            r = pymupdf.Rect(l["x0"] + 0.5, l["y0"] + 0.25 * h, l["x1"] - 0.5, l["y1"] - 0.25 * h)
            if not r.is_empty:
                mpage.add_redact_annot(r, fill=False)
                line_rects.append(pymupdf.Rect(l["x0"], l["y0"], l["x1"], l["y1"]))
    redacted = _apply_redactions(mpage)

    # 3) Notfall: Text ließ sich nicht entfernen (z. B. Scan mit Text-Ebene) oder liegt auf einem
    #    Bild -> mit der Hintergrundfarbe abdecken, damit nichts doppelt zu sehen ist
    for r in line_rects:
        still_there = False
        if redacted:
            for w in mpage.get_text("words", clip=r):
                if r.contains(pymupdf.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2)):
                    still_there = True
                    break
        if still_there or not redacted or any(not (r & ir).is_empty for ir in img_rects):
            bg, _fg = estimate_colors(arr, (r.x0 - 1) * zx, (r.y0 - 1) * zy, (r.x1 + 1) * zx, (r.y1 + 1) * zy)
            mpage.draw_rect(r, color=None, fill=tuple(c / 255 for c in bg), width=0, overlay=True)

    # 4) Übersetzung in Originalfarbe an dieselbe Stelle schreiben
    for b, font, (fs, placed) in queue:
        tw = pymupdf.TextWriter(mpage.rect)
        for x, y, t in placed:
            tw.append((x, y), t, font=font, fontsize=fs)
        tw.write_text(mpage, color=_rgb_from_int(b["color"]), overlay=True)
    return len(queue)


def overlay_untranslated_text(src_path, mono_path, out_path, src_lang, tgt_lang, translate_fn, log=print):
    """
    Findet Text, der in der von pdf2zh übersetzten PDF (mono_path) noch unverändert wie im
    Original (src_path) steht, übersetzt ihn mit translate_fn und setzt ihn an dieselbe Stelle.
    Ergebnis nach out_path, aber nur wenn mindestens ein Textblock ersetzt wurde.
    """
    stats = {"blocks": 0, "skipped": 0, "errors": 0}
    src = pymupdf.open(src_path)
    mono = pymupdf.open(mono_path)
    try:
        cache = {}
        for pno in range(min(len(src), len(mono))):
            if mono[pno].rotation != 0 or src[pno].rotation != 0:
                continue
            try:
                stats["blocks"] += _translate_page_text(src[pno], mono[pno], tgt_lang,
                                                       translate_fn, cache, stats, log)
            except Exception as e:  # eine problematische Seite darf den Rest nicht stoppen
                stats["errors"] += 1
                log(f"[Figures] Page {pno + 1} skipped: {type(e).__name__}: {e}")
        if stats["blocks"] > 0:
            try:
                mono.subset_fonts()
            except Exception:
                pass
            mono.save(out_path, garbage=3, deflate=True)
        return stats
    finally:
        src.close()
        mono.close()


def build_side_by_side(src_path, mono_path, out_path):
    """Zweisprachige PDF: Original links, Übersetzung rechts (je Seite)."""
    src = pymupdf.open(src_path)
    mono = pymupdf.open(mono_path)
    out = pymupdf.open()
    try:
        for i in range(min(len(src), len(mono))):
            r = src[i].rect
            page = out.new_page(width=r.width * 2, height=r.height)
            page.show_pdf_page(pymupdf.Rect(0, 0, r.width, r.height), src, i)
            page.show_pdf_page(pymupdf.Rect(r.width, 0, r.width * 2, r.height), mono, i)
        out.save(out_path, garbage=3, deflate=True)
    finally:
        out.close()
        src.close()
        mono.close()