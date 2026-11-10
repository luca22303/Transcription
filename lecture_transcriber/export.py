"""Export a finished session as Word, text, Markdown, subtitles or audio."""

from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

FORMATS: Dict[str, dict] = {
    "docx": {"label": "Word-Dokument", "ext": ".docx",
             "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
    "txt": {"label": "Text", "ext": ".txt", "mime": "text/plain; charset=utf-8"},
    "md": {"label": "Markdown (mit Zeitmarken)", "ext": ".md", "mime": "text/markdown; charset=utf-8"},
    "srt": {"label": "Untertitel (SRT)", "ext": ".srt", "mime": "application/x-subrip"},
    "wav": {"label": "Audioaufnahme (WAV)", "ext": ".wav", "mime": "audio/wav"},
}


def available_formats(session_dir: Path) -> List[str]:
    out = ["docx", "txt", "md", "srt"]
    if (session_dir / "aufnahme.wav").exists():
        out.append("wav")
    return out


def default_filename(info: dict, fmt: str) -> str:
    title = info.get("title") or "Vorlesung"
    started = info.get("started", "")
    try:
        date = datetime.fromisoformat(started).strftime("%Y-%m-%d")
    except ValueError:
        date = ""
    safe = "".join(c for c in title if c not in '/\\:*?"<>|').strip() or "Vorlesung"
    return f"{safe} {date}".strip() + FORMATS[fmt]["ext"]


def _date_line(info: dict) -> str:
    try:
        return datetime.fromisoformat(info.get("started", "")).strftime("%d.%m.%Y, %H:%M Uhr")
    except ValueError:
        return ""


def export_bytes(session_dir: Path, info: dict, paragraphs: List[dict], fmt: str) -> bytes:
    """Return the file content for the given format (not used for WAV)."""
    if fmt == "docx":
        return build_docx(info.get("title") or "Vorlesung", _date_line(info), paragraphs)
    if fmt == "txt":
        return "\n\n".join(p["text"] for p in paragraphs).encode("utf-8") + b"\n"
    if fmt == "md":
        src = session_dir / "transkript.md"
        return src.read_bytes() if src.exists() else b""
    if fmt == "srt":
        src = session_dir / "transkript.srt"
        return src.read_bytes() if src.exists() else b""
    raise ValueError(f"Unbekanntes Format: {fmt}")


def export_to(session_dir: Path, info: dict, paragraphs: List[dict], fmt: str, target: Path):
    target = Path(target)
    if not target.suffix:
        target = target.with_suffix(FORMATS[fmt]["ext"])
    if fmt == "wav":
        shutil.copyfile(session_dir / "aufnahme.wav", target)
    else:
        target.write_bytes(export_bytes(session_dir, info, paragraphs, fmt))
    return target


# ---------------------------------------------------------------------------
# Minimal .docx writer (stdlib only). Opens in Word, Pages and Google Docs.
# ---------------------------------------------------------------------------
_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults>
<w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/><w:sz w:val="23"/><w:lang w:val="de-DE"/></w:rPr></w:rPrDefault>
<w:pPrDefault><w:pPr><w:spacing w:after="160" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault>
</w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:pPr><w:spacing w:after="60"/></w:pPr><w:rPr><w:b/><w:sz w:val="44"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/>
<w:pPr><w:spacing w:after="360"/></w:pPr><w:rPr><w:color w:val="777777"/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>"""

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _run(text: str, props: str = "") -> str:
    rpr = f"<w:rPr>{props}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def build_docx(title: str, subtitle: str, paragraphs: List[dict],
               with_timestamps: Optional[bool] = True) -> bytes:
    body = [f'<w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr>{_run(title)}</w:p>']
    if subtitle:
        body.append(f'<w:p><w:pPr><w:pStyle w:val="Subtitle"/></w:pPr>{_run(subtitle)}</w:p>')
    for p in paragraphs:
        runs = ""
        if with_timestamps and p.get("time"):
            runs += _run(p["time"] + "  ", '<w:color w:val="999999"/><w:sz w:val="18"/>')
        runs += _run(p["text"])
        body.append(f"<w:p>{runs}</w:p>")
    body.append(
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1417" w:right="1417" w:bottom="1134" w:left="1417" '
        'w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>'
    )
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<w:document {_W}><w:body>{"".join(body)}</w:body></w:document>'
    )
    from io import BytesIO

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/_rels/document.xml.rels", _DOC_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", _STYLES)
    return buf.getvalue()
