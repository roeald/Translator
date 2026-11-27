"""
Eigenes Layout (ersetzt pdf2zh): Text absatzweise lesen, übersetzen und an dieselbe Stelle
setzen. Umbruch nur zwischen Wörtern, Fettdruck/Größe/Farbe bleiben, Bilder und Linien auch.
Nicht gedacht für Formelsatz; gedrehte Seiten bleiben unübersetzt.
"""
import re

import ocr_images as oi
from ocr_images import pymupdf

BULLETS = {"•", "◦", "▪", "‣", "●", "○", "■", "–", "-", "*", "", ""}
MIN_REL = 0.7  # so weit darf die Schrift schrumpfen, bevor es eng wird


def _is_bold(span):
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def _segments(page):
    """Textzeilen; weit auseinanderliegende Teile (Tabellenzellen, Fußzeile) bleiben getrennt."""
    spans = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1, 0))
            if dx > 0 and abs(dy) < 0.05:
                spans += [s for s in line["spans"] if s["text"].strip()]
    spans.sort(key=lambda s: (s["origin"][1], s["origin"][0]))
    rows = []
    for s in spans:
        if rows and abs(s["origin"][1] - rows[-1][0]["origin"][1]) <= 2:
            rows[-1].append(s)
        else:
            rows.append([s])

    segs = []
    for row in rows:
        row.sort(key=lambda s: s["origin"][0])
        groups, bullet = [], False
        for s in row:
            if s["text"].strip() in BULLETS and (not groups or s["bbox"][0] - groups[-1]["x1"] > 8):
                bullet = True  # Aufzählungszeichen bleibt unangetastet stehen
                continue
            x0, x1 = s["bbox"][0], s["bbox"][2]
            if groups and not bullet and x0 - groups[-1]["x1"] <= max(1.5 * s["size"], 8):
                g = groups[-1]
                g["gaps"].append(x0 - g["x1"] > 1)
                g["spans"].append(s)
                g["x1"] = max(g["x1"], x1)
            else:
                groups.append({"spans": [s], "gaps": [], "x0": x0, "x1": x1, "bullet": bullet})
                bullet = False
        for g in groups:
            sp = g["spans"]
            text = sp[0]["text"]
            for s, gap in zip(sp[1:], g["gaps"]):
                text += (" " if gap else "") + s["text"]
            main = max(sp, key=lambda s: len(s["text"]))
            chars = sum(len(s["text"].strip()) for s in sp)
            bold_chars = sum(len(s["text"].strip()) for s in sp if _is_bold(s))
            lead = ""
            if _is_bold(sp[0]) and bold_chars < 0.9 * chars:  # fetter Einstieg ("Option A: ...")
                for s in sp:
                    if not _is_bold(s):
                        break
                    lead += s["text"]
            g.update(text=" ".join(text.split()), lead=" ".join(lead.split()),
                     y0=min(s["bbox"][1] for s in sp), y1=max(s["bbox"][3] for s in sp),
                     base=main["origin"][1], size=main["size"], color=main["color"],
                     bold=bold_chars > 0.6 * chars,
                     math=any(k in main["font"].lower() for k in oi.FORMULA_FONTS))
            segs.append(g)
    return segs


def _column_right(seg, segs):
    return max(s["x1"] for s in segs if abs(s["x0"] - seg["x0"]) <= 3) if segs else seg["x1"]


def _continues(par, seg, segs, font):
    """Ist seg die Fortsetzung des Absatzes par?"""
    last = par["lines"][-1]
    d = seg["base"] - last["base"]
    if seg["bullet"] or abs(seg["size"] - last["size"]) > 0.6 or not 0 < d <= 1.9 * seg["size"]:
        return False
    indent = len(par["lines"]) == 1 and 0 < last["x0"] - seg["x0"] <= 40  # Erstzeileneinzug
    centered = abs((seg["x0"] + seg["x1"]) - (last["x0"] + last["x1"])) <= 4
    if abs(seg["x0"] - last["x0"]) > 2 and not indent and not centered:
        return False
    if len(par["lines"]) > 1 and abs(d - (last["base"] - par["lines"][-2]["base"])) > 2:
        return False
    if seg["bold"] != last["bold"] and not last["lead"]:
        return False
    prev = last["text"]
    if seg["text"][:1].islower() or prev.endswith(("-", ",")):
        return True
    if prev.endswith(":"):
        return False
    # Hätte das erste Wort noch in die vorige Zeile gepasst, war der Umbruch Absicht
    first = seg["text"].split()[0]
    need = font.text_length(" " + first, fontsize=seg["size"])
    return last["x1"] + need > _column_right(last, segs) - 2


def _paragraphs(segs, font):
    pars = []
    for s in sorted(segs, key=lambda s: (s["base"], s["x0"])):
        for p in reversed(pars[-40:]):
            if _continues(p, s, segs, font):
                p["lines"].append(s)
                break
        else:
            pars.append({"lines": [s]})
    for p in pars:
        ls = p["lines"]
        p.update(x0=min(l["x0"] for l in ls), x1=max(l["x1"] for l in ls),
                 y0=min(l["y0"] for l in ls), y1=max(l["y1"] for l in ls),
                 text=oi._join_lines([l["text"] for l in ls]), size=ls[0]["size"],
                 color=ls[0]["color"], bold=ls[0]["bold"], lead=ls[0]["lead"])
    return pars


def _worth(p):
    t = p["text"]
    if sum(c.isalpha() for c in t) < 2 or any(l["math"] for l in p["lines"]):
        return False
    return not (" " not in t and re.search(r"https?://|www\.|\S@\S", t))


def _translate(p, translate_fn):
    """Liste von (Text, fett). Fetter Einstieg und '|'-Teile werden einzeln übersetzt."""
    def tr(t):
        return " ".join("".join(
            part if i % 2 else (str(translate_fn(part)) if part.strip() else part)
            for i, part in enumerate(re.split(r"(\s\|\s)", t))).split())

    lead = p["lead"]
    if lead and p["text"].startswith(lead) and p["text"][len(lead):].strip():
        return [(tr(lead), True), (tr(p["text"][len(lead):].strip()), False)]
    return [(tr(p["text"]), p["bold"])]


def _wrap(runs, tgt_lang, fs, width):
    """Umbruch nach Wörtern. Ergebnis: Zeilen aus [(Wort, Font, x-Versatz)], Breiten, passt?"""
    no_space = tgt_lang in oi.NO_SPACE_LANGS
    lines, widths, cur, w, ok = [], [], [], 0.0, True
    for text, bold in runs:
        font = oi._load_font(tgt_lang, bold)
        for word in (list(text) if no_space else text.split()):
            ww = font.text_length(word, fontsize=fs)
            gap = 0.0 if no_space or not cur else font.text_length(" ", fontsize=fs)
            if cur and w + gap + ww > width:
                lines.append(cur)
                widths.append(w)
                cur, w, gap = [], 0.0, 0.0
            cur.append((word, font, w + gap))
            w += gap + ww
            ok = ok and ww <= width + 0.01
    if cur:
        lines.append(cur)
        widths.append(w)
    return lines, widths, ok


def _layout(p, runs, tgt_lang, align, L, R, bottom):
    n, size = len(p["lines"]), p["size"]
    cx = (p["x0"] + p["x1"]) / 2
    width = {0: R - p["x0"], 1: 2 * min(cx - L, R - cx), 2: p["x1"] - L}[align]
    width = max(width, p["x1"] - p["x0"])
    bases = [l["base"] for l in p["lines"]]
    pitch = (bases[-1] - bases[0]) / (n - 1) if n > 1 else 1.2 * size
    fs = size
    while True:
        lines, widths, ok = _wrap(runs, tgt_lang, fs, width)
        step = pitch
        fits = len(lines) <= n
        if not fits and (n > 1 or fs <= MIN_REL * size):  # zusätzliche Zeilen in den Platz darunter
            step = min(pitch, (bottom - bases[0] - 0.25 * fs) / (len(lines) - 1))
            fits = step >= 1.15 * fs
        if fs <= 4:
            step = max(step, fs)
        if (ok and fits) or fs <= 4:
            break
        fs = max(4.0, fs * 0.96)  # passt noch nicht: etwas kleiner setzen
    out = []
    for i, (words, lw) in enumerate(zip(lines, widths)):
        x = p["x0"] if align == 0 else (cx - lw / 2 if align == 1 else p["x1"] - lw)
        out += [((x + off, bases[0] + i * step), word, font) for word, font, off in words]
    return fs, out


def _translate_page(page, tgt_lang, translate_fn, cache, log):
    segs = _segments(page)
    if not segs:
        return 0
    pars = _paragraphs(segs, oi._load_font(tgt_lang))
    page_w, page_h = page.rect.width, page.rect.height
    left_margin = sorted(s["x0"] for s in segs)[len(segs) // 4]
    right_margin = max(s["x1"] for s in segs)
    arr, pw, ph = oi._render_region(page, page.rect)
    zx, zy = pw / page_w, ph / page_h

    queue = []
    for p in pars:
        if not _worth(p):
            continue
        key = (p["text"], p["lead"])
        if key not in cache:
            cache[key] = _translate(p, translate_fn)
        runs = cache[key]
        if oi._norm(" ".join(t for t, _ in runs)) == oi._norm(p["text"]):
            continue  # unverändert: Original stehen lassen
        align = oi._block_align_pt(p, pars, page_w, left_margin)
        if len(p["lines"]) == 1 and p["x0"] > page_w / 2 and abs(p["x1"] - right_margin) <= 3:
            align = 2  # rechtsbündig, z. B. "Seite 5 von 17"
        if len(p["lines"]) > 1:
            L, R = p["x0"], max(p["x1"], _column_right(p["lines"][0], segs))
        else:
            L, _a, R, _b = oi._text_box_px(p, pars, max(right_margin, p["x1"]), align, min(left_margin, p["x0"]))
            if align == 0:
                R = max(right_margin, p["x1"])
            bg, _fg = oi.estimate_colors(arr, p["x0"] * zx, p["y0"] * zy, p["x1"] * zx, p["y1"] * zy)
            L, R = oi._clip_to_free_space(arr, zx, zy, p, L, R, bg)
        below = [s["y0"] for s in segs if s["y0"] >= p["y1"] - 1 and s["x0"] < R and s["x1"] > L]
        bottom = min(below, default=page_h - 20) - 0.9 * p["size"]  # Abstand zum Nächsten lassen
        queue.append((p, _layout(p, runs, tgt_lang, align, L, R, bottom)))
    if not queue:
        return 0

    for p, _l in queue:
        for l in p["lines"]:
            for s in l["spans"]:
                x0, y0, x1, y1 = s["bbox"]
                h = y1 - y0
                r = pymupdf.Rect(x0 + 0.3, y0 + 0.3 * h, x1 - 0.3, y1 - 0.3 * h)
                if not r.is_empty:
                    page.add_redact_annot(r, fill=False)
    if not oi._apply_redactions(page):
        raise RuntimeError("Original text could not be removed (PyMuPDF too old).")

    for p, (fs, placed) in queue:
        tw = pymupdf.TextWriter(page.rect)
        for pos, word, font in placed:
            tw.append(pos, word, font=font, fontsize=fs)
        tw.write_text(page, color=oi._rgb_from_int(p["color"]))
    return len(queue)


def translate_layout(in_path, out_path, tgt_lang, translate_fn, log=print):
    """Übersetzt den echten Text der PDF. Gibt Statistik zurück; out_path wird immer geschrieben."""
    doc = pymupdf.open(in_path)
    try:
        stats, cache = {"blocks": 0, "rotated": 0, "failed": 0}, {}
        for pno, page in enumerate(doc):
            log(f"[Layout] Page {pno + 1}/{len(doc)}")
            if page.rotation != 0:
                stats["rotated"] += 1
                continue
            try:
                stats["blocks"] += _translate_page(page, tgt_lang, translate_fn, cache, log)
            except Exception as e:  # eine problematische Seite darf den Rest nicht stoppen
                stats["failed"] += 1
                log(f"[Layout] Page {pno + 1} skipped: {type(e).__name__}: {e}")
        try:
            doc.subset_fonts()
        except Exception:
            pass
        doc.save(out_path, garbage=3, deflate=True)
        return stats
    finally:
        doc.close()