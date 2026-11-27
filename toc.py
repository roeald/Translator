"""
Inhaltsverzeichnisse (Zeilen mit Punktlinie und Seitenzahl).

pdf2zh klebt solche Zeilen zu riesigen Texten voller Punkte zusammen. Das Sprachmodell
gerät dabei in Wiederholungsschleifen (die App scheint zu hängen). Deshalb:
  1) strip_toc():  Zeilen merken und aus der Eingabe für pdf2zh entfernen
  2) draw_toc():   nur die Titel übersetzen und Zeile für Zeile neu setzen
"""
import re

import ocr_images
from ocr_images import pymupdf

ROW = re.compile(
    r"^(?P<title>.*?[^.\s…·].*?)\s*(?P<leader>(?:\.\s?){4,}|…{2,}|(?:·\s?){4,})\s*"
    r"(?P<page>\d{1,4}|[ivxlIVXL]{1,6})\s*$"
)
NUMBER = re.compile(r"^((?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+)(?=\S)")  # "1 ", "2.1 ", "A. "
MIN_ROWS = 3  # erst ab so vielen Treffern gilt eine Seite als Verzeichnis


def _rows(page):
    """Alle waagerechten Textstücke nach Grundlinie zu Zeilen zusammensetzen."""
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
    return [sorted(r, key=lambda s: s["origin"][0]) for r in rows]


def _find_entries(page):
    entries = []
    for row in _rows(page):
        text = " ".join("".join(s["text"] for s in row).split())
        m = ROW.match(text)
        if not m:
            continue
        main = max(row, key=lambda s: len(s["text"]))
        entries.append({
            "title": m["title"], "page": m["page"],
            "x0": min(s["bbox"][0] for s in row), "x1": max(s["bbox"][2] for s in row),
            "base": main["origin"][1], "size": main["size"], "color": main["color"],
            "bold": bool(main["flags"] & 16) or "bold" in main["font"].lower(),
            "boxes": [tuple(s["bbox"]) for s in row],
        })
    return entries if len(entries) >= MIN_ROWS else []


def strip_toc(src_path, out_path):
    """Gibt {Seitennummer: Einträge} zurück. out_path wird nur geschrieben, wenn etwas gefunden wurde."""
    doc = pymupdf.open(src_path)
    try:
        found = {}
        for pno, page in enumerate(doc):
            if page.rotation != 0:
                continue
            entries = _find_entries(page)
            if not entries:
                continue
            for e in entries:
                for x0, y0, x1, y1 in e["boxes"]:
                    h = y1 - y0
                    page.add_redact_annot(pymupdf.Rect(x0, y0 + 0.3 * h, x1, y1 - 0.3 * h), fill=False)
            if ocr_images._apply_redactions(page):
                found[pno] = entries
        if found:
            doc.save(out_path, garbage=3, deflate=True)
        return found
    finally:
        doc.close()


def draw_toc(mono_path, out_path, found, tgt_lang, translate_fn, log=print):
    """Schreibt die übersetzten Einträge auf die von pdf2zh erzeugte PDF."""
    doc = pymupdf.open(mono_path)
    try:
        count, total = 0, sum(len(v) for v in found.values())
        for pno, entries in found.items():
            page = doc[pno]
            for e in entries:
                count += 1
                log(f"[TOC] Entry {count}/{total}")
                m = NUMBER.match(e["title"])
                number = m.group(1) if m else ""
                title = e["title"][len(number):]
                try:
                    translated = " ".join(str(translate_fn(title)).split()).rstrip(" .…") or title
                except Exception as exc:  # lieber das Original zeigen als eine Lücke
                    log(f"[TOC] Kept original for {title!r}: {exc}")
                    translated = title
                font = ocr_images._load_font(tgt_lang, e["bold"])
                label = number + translated
                fs = e["size"]
                width = e["x1"] - e["x0"]
                # Titel + mindestens 6 Punkte + Seitenzahl müssen in die Zeile passen
                need = font.text_length(label + "......" + e["page"], fontsize=fs) + 2
                if need > width:
                    fs *= width / need
                page_x = e["x1"] - font.text_length(e["page"], fontsize=fs)
                dot = font.text_length(".", fontsize=fs)
                dots = max(4, int((page_x - e["x0"] - font.text_length(label, fontsize=fs) - 1) / dot))
                tw = pymupdf.TextWriter(page.rect)
                tw.append((e["x0"], e["base"]), label + "." * dots, font=font, fontsize=fs)
                tw.append((page_x, e["base"]), e["page"], font=font, fontsize=fs)
                tw.write_text(page, color=ocr_images._rgb_from_int(e["color"]))
        try:
            doc.subset_fonts()
        except Exception:
            pass
        doc.save(out_path, garbage=3, deflate=True)
        return count
    finally:
        doc.close()