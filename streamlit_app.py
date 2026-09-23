"""
ISO Cut Length Extractor - prototype
------------------------------------
Uploads a multi-page piping isometric PDF, OCRs the CUT PIPE LENGTH table,
and exports an Excel register.

Requirements:
    pip install -r requirements.txt

System dependency:
    Tesseract OCR must be installed and available as "tesseract" on PATH.
"""

import io
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import fitz
import cv2
import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------
# OCR helpers
# ----------------------------

def run_tesseract(image: np.ndarray, psm: int = 6, tsv: bool = True, timeout: int = 8) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    cmd = ["tesseract", "stdin", "stdout", "--psm", str(psm)]
    if tsv:
        cmd.append("tsv")
    try:
        result = subprocess.run(
            cmd,
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return result.stdout.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def tsv_tokens(tsv: str):
    tokens = []
    lines = tsv.splitlines()
    if len(lines) <= 1:
        return tokens

    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        text = parts[11].strip()
        if not text:
            continue
        try:
            conf = float(parts[10])
        except Exception:
            conf = -1
        try:
            x, y, w, h = map(int, parts[6:10])
        except Exception:
            continue
        tokens.append({
            "text": text,
            "x": x, "y": y, "w": w, "h": h,
            "conf": conf
        })
    return tokens


def render_clip(page, x0, y0, x1, y1, scale=4):
    r = page.rect
    clip = fitz.Rect(x0*r.width, y0*r.height, x1*r.width, y1*r.height)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False)
    return cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_GRAYSCALE)


# ----------------------------
# Drawing number
# ----------------------------

def extract_drawing_number(page, fallback=None):
    img = render_clip(page, 0.67, 0.83, 0.995, 0.995, scale=2)
    text = run_tesseract(img, psm=11, tsv=False, timeout=6)

    # The drawing number in this drawing family is the final 3-digit number
    # in the title-block OCR.
    nums = re.findall(r"(?<!\d)(\d{3})(?!\d)", text)
    if nums:
        return nums[-1], text

    return fallback, text


# ----------------------------
# Cut table extraction
# ----------------------------

def extract_cut_candidates(page):
    """
    The uploaded drawing family places CUT PIPE LENGTH in the upper-right.
    We render a deliberately generous region because tables can have
    different numbers of cut rows.
    """
    img = render_clip(page, 0.54, 0.02, 0.73, 0.25, scale=4)

    # Run two OCR variants. The second is useful on faint/clouded tables.
    gray_tsv = run_tesseract(img, psm=6, tsv=True, timeout=8)

    # Adaptive threshold for a second opinion.
    adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    adaptive_tsv = run_tesseract(adaptive, psm=6, tsv=True, timeout=8)

    all_candidates = []

    for source, tsv in [("gray", gray_tsv), ("adaptive", adaptive_tsv)]:
        for t in tsv_tokens(tsv):
            if not re.fullmatch(r"\d{2,5}", t["text"]):
                continue

            value = int(t["text"])
            # Exclude header / piece numbers / revision marks.
            if value < 20:
                continue

            # In this crop the cut-length column is around the middle-right.
            relx = t["x"] / max(img.shape[1], 1)
            rely = t["y"] / max(img.shape[0], 1)

            if not (0.32 <= relx <= 0.72):
                continue
            if rely < 0.14:
                continue

            all_candidates.append({
                "value": value,
                "y": t["y"] + t["h"]/2,
                "conf": max(0.0, t["conf"]),
                "source": source,
            })

    # Cluster candidates by vertical position.
    all_candidates.sort(key=lambda x: x["y"])
    clusters = []
    for c in all_candidates:
        if not clusters or abs(c["y"] - clusters[-1][-1]["y"]) > 24:
            clusters.append([c])
        else:
            clusters[-1].append(c)

    rows = []
    for cluster in clusters:
        # Prefer agreement between OCR passes. If both passes agree,
        # confidence is boosted; if they disagree, flag for review.
        values = defaultdict(list)
        for c in cluster:
            values[c["value"]].append(c)

        ranked = sorted(
            values.items(),
            key=lambda kv: (
                len(kv[1]),
                max(x["conf"] for x in kv[1])
            ),
            reverse=True
        )
        value, matches = ranked[0]
        best_conf = max(x["conf"] for x in matches)
        agreement = len(matches) >= 2
        disagreement = len(values) > 1

        rows.append({
            "length": value,
            "ocr_confidence": round(best_conf, 1),
            "agreement": agreement,
            "review": (not agreement) or disagreement or best_conf < 75,
            "y": sum(x["y"] for x in cluster) / len(cluster),
        })

    # Deduplicate nearby rows if OCR generated two clusters for one row.
    cleaned = []
    for row in rows:
        if cleaned and abs(row["y"] - cleaned[-1]["y"]) < 28:
            old = cleaned[-1]
            if row["ocr_confidence"] > old["ocr_confidence"]:
                cleaned[-1] = row
        else:
            cleaned.append(row)

    return cleaned


def process_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    records = []
    page_summary = []

    for page_index in range(len(doc)):
        page = doc[page_index]

        # Page-number fallback is only a fallback; title-block OCR is preferred.
        fallback_drawing = str(106 + page_index + 1)
        drawing, title_ocr = extract_drawing_number(page, fallback=fallback_drawing)

        rows = extract_cut_candidates(page)

        # Piece letters are sequential in the CUT PIPE LENGTH table in this
        # drawing family. We assign A/B/C... in detected row order.
        for i, row in enumerate(rows):
            cut_id = chr(ord("A") + i) if i < 26 else f"R{i+1}"

            records.append({
                "Drawing No.": drawing,
                "Cut ID": cut_id,
                "Cut Length (mm)": row["length"],
                "PDF Page": page_index + 1,
                "OCR Confidence": row["ocr_confidence"],
                "OCR Agreement": "YES" if row["agreement"] else "NO",
                "Review": "REVIEW" if row["review"] else "OK",
            })

        page_summary.append({
            "PDF Page": page_index + 1,
            "Drawing No.": drawing,
            "Cuts Detected": len(rows),
            "Status": "REVIEW" if any(r["review"] for r in rows) else "OK",
        })

    return pd.DataFrame(records), pd.DataFrame(page_summary)


def make_excel(cuts_df, pages_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cuts_df.to_excel(writer, index=False, sheet_name="Cut Register")
        pages_df.to_excel(writer, index=False, sheet_name="Page Review")

        wb = writer.book

        for ws in [wb["Cut Register"], wb["Page Review"]]:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for cell in ws[1]:
                cell.font = Font(bold=True)

            for col in range(1, ws.max_column + 1):
                max_len = 0
                for cell in ws[get_column_letter(col)]:
                    max_len = max(max_len, len(str(cell.value or "")))
                ws.column_dimensions[get_column_letter(col)].width = min(max_len + 2, 28)

    output.seek(0)
    return output.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(
    page_title="ISO Cut Length Extractor",
    page_icon="📐",
    layout="wide"
)

st.title("📐 ISO Cut Length Extractor")
st.caption("Piping Isometric PDF → CUT ID + Cut Length → Excel")

uploaded = st.file_uploader(
    "Upload the complete isometric PDF",
    type=["pdf"]
)

if uploaded:
    st.info(
        "Prototype mode: the extractor is designed around the drawing layout in "
        "the supplied TRAIL PDF. Always review records marked REVIEW before using "
        "the Excel for fabrication/MTO purposes."
    )

    if st.button("Extract Cut Lengths", type="primary"):
        with st.spinner("Processing drawings..."):
            cuts, pages = process_pdf(uploaded.getvalue())
            excel = make_excel(cuts, pages)

        st.success(f"Processed {len(pages)} PDF pages and detected {len(cuts)} cut rows.")

        c1, c2, c3 = st.columns(3)
        c1.metric("PDF pages", len(pages))
        c2.metric("Cuts detected", len(cuts))
        c3.metric("Rows for review", int((cuts["Review"] == "REVIEW").sum()))

        st.subheader("Cut Register Preview")
        st.dataframe(cuts, use_container_width=True, height=450)

        st.download_button(
            "⬇️ Download Excel",
            data=excel,
            file_name="ISO_Cut_Length_Register.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.subheader("Page Review")
        st.dataframe(pages, use_container_width=True)
