"""Assemble final_package/ for the Quantitative Finance submission and verify it.

Every file in final_package/ is produced here from the audited manuscript tree, so
the package can be rebuilt byte-for-byte after any upstream change.  Run from the
repository root:  python -B scripts/build_final_package.py
Needs pdflatex, bibtex and python-docx (the .docx deliverables).
"""
from pathlib import Path
import datetime, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "manuscript"
PKG = ROOT / "final_package"
STAMP = (2026, 11, 25, 0, 0, 0)          # fixed zip timestamp keeps archives reproducible
DATE = "25 November 2026"
TITLE = ("RoughCert: Machine-Verified VIX Derivative Bounds and Conditional Risk "
         "Certificates under Rough Volatility")
JOURNAL = "Quantitative Finance"
REPO = "https://github.com/EslamAhmedGaber/RoughCert"
TAG = "v5.0.0"

HIGHLIGHTS = [
    "Deterministic 169-bit Arb brackets certify continuous-time VIX futures and options",
    "The largest certified futures separation exceeds 1.62 standard Cboe ticks",
    "Splitting at the exercise boundary exposes downward IV bias at all seven strikes",
    "Direct IV expansion falls below the certified band at 52 of 60 literature strikes",
    "Residual variance share distinguishes tight VIX bounds from wide spot-risk brackets",
]

COVER_BULLETS = [
    ("Certified continuous-time prices.", "An exact Gaussian Mehler error ledger and two outward-rounded 169-bit Arb engines "
     "enclose continuous-time VIX futures, call prices and Black-76 implied volatilities. The verdicts need no simulated continuum reference."),
    ("Where a fast expansion breaks the tick.", "At five high vol-of-vol cells the order-three weak futures expansion of Bourgey, De Marco and Gobet (2023) "
     "exceeds the certified continuous-time price by more than one Cboe tick (0.05 points), with a certified separation of at least 1.62 ticks in the largest case."),
    ("Implied-volatility bias.", "On a mixed rough Bergomi smile, splitting the expansion integral at the exercise boundary exposes a downward "
     "implied-volatility bias at all seven strikes, and the direct expansion of Liao, Agarwal and Bourgey (2026) falls below the certified band at 52 of 60 published strikes."),
    ("Cost of Markovian and quantisation shortcuts.", "An optimised Markovian reduction needs 14 factors for sub-tick futures at $H=0.05$, and a 960-path "
     "product functional quantiser underprices all 18 literature futures by at least 0.208 points."),
    ("Conditional risk with explicit labels.", "Integrated-variance brackets remain statistical and are labelled as such; the residual variance share "
     "helps explain why the tested futures enclosures are narrower than the spot-risk brackets."),
]


def team_citation_title():
    """Title of the mandatory team citation, read from the bibliography so it cannot drift."""
    import re
    bib = (M / "references.bib").read_text(encoding="utf-8", errors="replace")
    entry = bib[bib.index("@article{AHMED2026111045,"):]
    title = re.search(r"title\s*=\s*\{(.+?)\},\n", entry, re.S).group(1)
    return re.sub(r"[{}]", "", title).replace("--", "\u2013")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_zip(target, items, comment=""):
    """items: list of (source path, archive name), written in sorted order."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for src, arc in sorted(items, key=lambda t: t[1]):
            info = zipfile.ZipInfo(arc, date_time=STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, Path(src).read_bytes())
        if comment:
            z.comment = comment.encode()
    return target


def latex_sources():
    items = []
    for name in ["main.tex", "one_column.tex", "two_columns.tex", "references.bib", "main.bbl"]:
        items.append((M / name, name))
    for sub in ["sections", "tables", "figures"]:
        for p in sorted((M / sub).iterdir()):
            if p.suffix in {".tex", ".pdf"}:
                items.append((p, f"{sub}/{p.name}"))
    for name in ["supplement.tex", "supplement.bbl"]:
        items.append((M / "supplement" / name, f"supplement/{name}"))
    readme = (
        "RoughCert - LaTeX source\n"
        "========================\n\n"
        "main.tex is the master; it compiles directly as the one-column edition.\n"
        "Compile one_column.tex or two_columns.tex for the two submission editions,\n"
        "and supplement/supplement.tex for the supplementary material.\n\n"
        "  pdflatex one_column   bibtex one_column   pdflatex one_column (x2)\n"
        "  pdflatex two_columns  bibtex two_columns  pdflatex two_columns (x2)\n"
        "  cd supplement && pdflatex supplement && bibtex supplement && pdflatex supplement (x2)\n\n"
        "Requires the elsarticle class and the packages listed in the preamble of main.tex;\n"
        "The master and supplement bibliographies are included. Run BibTeX for either wrapper.\n"
        "Figures are vector PDFs regenerated by scripts/build_v3_figures.py in the code archive.\n"
    )
    tmp = PKG / "_README_SOURCE.txt"
    tmp.write_text(readme, encoding="utf-8", newline="\n")
    items.append((tmp, "README_SOURCE.txt"))
    return items


def code_sources():
    base = ROOT / "public_repo" if (ROOT / "public_repo").is_dir() else ROOT
    paths = (base / "verification/release_files.txt").read_text(encoding="utf-8").splitlines()
    return [(base / rel, rel) for rel in paths]


def write_highlights():
    (PKG / "Highlights.txt").write_text("".join("- " + h + "\n" for h in HIGHLIGHTS),
                                        encoding="utf-8", newline="\n")
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)
    head = doc.add_paragraph()
    head.add_run("Highlights").bold = True
    doc.add_paragraph(TITLE)
    for h in HIGHLIGHTS:
        doc.add_paragraph(h, style="List Bullet")
    doc.save(PKG / "Highlights.docx")


def cover_letter_text():
    """Returns (list of paragraphs, list of (lead, body)) in plain text for the docx."""
    intro = (f"I am writing on behalf of the authors to submit our manuscript entitled \u201c{TITLE}\u201d "
             f"for consideration as a research article in {JOURNAL}.")
    motive = ("Fast VIX approximations under rough volatility are validated against simulation, so a desk learns the "
              "sampling noise of a discretised target rather than the distance to the continuous-time price. "
              "For readers of Quantitative Finance who price or validate volatility derivatives, RoughCert measures this error against a continuous-time reference in exchange ticks.")
    tail = [
        ("The supplementary material gives the measure-axis rate study, the coverage diagnostics and the layered certificate "
         f"summary. The code, the stored numerical results and the scripts and archived inputs for every table, figure and reported "
         f"number are available at {REPO} (release {TAG})."),
        ("Related work by the authors: \u201c" + team_citation_title() + "\u201d, "
         "Communications in Nonlinear Science and Numerical Simulation 2026, doi:10.1016/j.cnsns.2026.111045."),
        ("CRediT authorship contributions, funding, competing interests and generative-AI use are disclosed in the manuscript, "
         "which is not under consideration elsewhere."),
        "Thank you for considering our manuscript.",
    ]
    return [intro, motive], tail


def write_cover_letter():
    folder = PKG / "cover_letter"
    folder.mkdir(parents=True, exist_ok=True)
    head, tail = cover_letter_text()
    bullets = "\n".join(r"\item \textbf{%s} %s" % (lead, body) for lead, body in COVER_BULLETS)
    tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.9cm,top=1.5cm,bottom=1.5cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{tgtermes}
\usepackage{enumitem}\setlist[itemize]{label=\textbullet,leftmargin=1.6em,labelsep=0.55em,itemsep=1pt,topsep=1pt,parsep=0pt,partopsep=0pt}
\usepackage[hidelinks]{hyperref}\urlstyle{same}
\setlength{\parindent}{0pt}\setlength{\parskip}{3.5pt}\pagestyle{empty}\emergencystretch=2em
\newcommand{\ReleaseTag}{RELEASE_TAG}
\begin{document}
\textbf{Eslam Ahmed}\\
Corresponding author\\
Department of Engineering Mathematics and Physics\\Faculty of Engineering, Cairo University, Giza 12613, Egypt\\
\href{mailto:eslamahmed@eng.cu.edu.eg}{eslamahmed@eng.cu.edu.eg}
\par\medskip
\hfill DATE
\par\medskip
\textbf{The Editors}\\
\emph{JOURNAL}
\par\medskip
\textbf{Dear Editors,}

INTRO

MOTIVE
\begin{itemize}
BULLETS
\end{itemize}

TAIL1

TAIL2

TAIL3

TAIL4

\textbf{Sincerely,}\\
\textbf{Eslam Ahmed}\\
On behalf of the authors
\end{document}
"""
    tex = (tex.replace("DATE", DATE).replace("JOURNAL", JOURNAL)
              .replace("INTRO", head[0]).replace("MOTIVE", head[1]).replace("BULLETS", bullets))
    for i, t in enumerate(tail, start=1):
        tex = tex.replace(f"TAIL{i}", t)
    tex = tex.replace("(release " + TAG + ")", r"(release \href{" + REPO + "/releases/tag/" + TAG + r"}{\ReleaseTag})").replace("RELEASE_TAG", TAG)
    tex = tex.replace("\u201c", "``").replace("\u201d", "''")
    name = "QF_cover_letter"
    (folder / (name + ".tex")).write_text(tex, encoding="utf-8", newline="\n")
    env = {**os.environ, "max_print_line": "10000"}
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                       cwd=folder, capture_output=True, env=env, check=True)
    import re
    log = (folder / (name + ".log")).read_text(errors="replace")
    if re.search(r'Overfull \\[hv]box|Underfull \\hbox|^!|LaTeX Warning: (?:Reference|Citation)',log,re.M):
        raise RuntimeError("Cover-letter TeX diagnostics require review")
    for junk in [".aux", ".log", ".out"]:
        (folder / (name + junk)).unlink(missing_ok=True)
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(11)
    from docx.shared import Inches
    section = doc.sections[0]
    section.page_height, section.page_width = Inches(11.69), Inches(8.27)
    section.top_margin = section.bottom_margin = Inches(0.70)
    section.left_margin = section.right_margin = Inches(0.78)
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Normal"].paragraph_format.space_after = Pt(2)
    doc.styles["Normal"].paragraph_format.line_spacing = 1.0
    for line in ["Eslam Ahmed", "Corresponding author",
                 "Department of Engineering Mathematics and Physics",
                 "Faculty of Engineering, Cairo University, Giza 12613, Egypt",
                 "eslamahmed@eng.cu.edu.eg", "", DATE, "", "The Editors", JOURNAL, "", "Dear Editors,"]:
        doc.add_paragraph(line)
    doc.add_paragraph(head[0])
    doc.add_paragraph(head[1])
    for lead, body in COVER_BULLETS:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(lead + " ").bold = True
        p.add_run(body.replace("$H=0.05$", "H = 0.05"))
    for t in tail:
        doc.add_paragraph(t)
    for line in ["", "Sincerely,", "Eslam Ahmed", "On behalf of the authors"]:
        doc.add_paragraph(line)
    doc.save(folder / (name + ".docx"))
    return (folder / (name + ".pdf")).is_file()


def write_readme(files, room):
    text = f"""# RoughCert \u2014 submission package

{TITLE}

Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman, Sroor M. Elnady \u2014 prepared for *{JOURNAL}*. Public code and data: [{TAG}]({REPO}/releases/tag/{TAG}).

## Contents

| File | What it is |
| --- | --- |
| `RoughCert_one_column.pdf` | Manuscript, one-column edition (review copy, {room["one_column"]["pages"]} pages). |
| `RoughCert_two_columns.pdf` | Manuscript, two-column edition (journal-style preview, {room["two_columns"]["pages"]} pages). |
| `RoughCert_supplementary_material.pdf` | Supplementary material ({room["supplement"]["pages"]} pages). |
| `RoughCert_LaTeX_source.zip` | Complete LaTeX source of both editions and the supplement, with figures and `.bbl` files. |
| `RoughCert_code_and_data.zip` | Arb pricing engines, comparison drivers, stored results and the scripts and archived inputs for every table, figure and reported number. |
| `Highlights.txt`, `Highlights.docx` | Five highlights, each at most 85 characters. |
| `cover_letter/` | Cover letter (LaTeX source, PDF, DOCX). |
| `SHA256SUMS.txt`, `SHA256_MANIFEST.json` | SHA-256 of every file in this package. |

## Editions

Both PDFs come from one master, `main.tex`. Compiling `main.tex` or `one_column.tex` gives the
one-column edition; `two_columns.tex` gives the two-column edition, in which all floats span the
page width. The text, numbering, tables, figures and references are identical in the two editions.

## Reproducing the numbers

Unpack `RoughCert_code_and_data.zip` and run:

```
pip install -r requirements.txt
python reproduce.py
```

The display pipeline reads the recorded 169-bit Arb enclosures, rewrites the generated macro
files, rebuilds the ten tables and nine figures from the stored JSON results, and runs the audit
scripts that check every printed number against its source. Appendix B of the manuscript lists the
experiments, the fixed parameters and the recorded software versions.

## Verification gates

The manuscript tree passes four gates, all with exit code 0:

- `build_verify.py` \u2014 compiles the one-column edition, the two-column edition and the supplement,
  then runs the writing-norm, LaTeX-hygiene, technical-compliance and leak scans.
- `audit_v3_tables.py` \u2014 recomputes all 1,175 printed numeric entries from the stored results with the
  declared outward rounding.
- `audit_v3_figures.py` \u2014 checks the figure data, labels and annotations against the same results.
- `audit_v3_release.py` \u2014 checks the prose numbers, macro definitions, cross-references, the proof
  edit ledger and both PDFs.

## Files

{files}
"""
    (PKG / "README.md").write_text(text, encoding="utf-8", newline="\n")


def clean_room(zip_path):
    """Extract the source archive into an empty directory and rebuild all three documents."""
    import re, fitz
    report = {}
    room_parent = ROOT / "audit"
    room_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cleanroom_", dir=room_parent) as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        pdfs = {"one_column": "RoughCert_one_column.pdf", "two_columns": "RoughCert_two_columns.pdf",
                "supplement": "RoughCert_supplementary_material.pdf"}
        for stem, target in pdfs.items():
            folder = tmp / "supplement" if stem == "supplement" else tmp
            tex = ["pdflatex", "-halt-on-error", "-interaction=nonstopmode", "-file-line-error", stem + ".tex"]
            phases = [tex, ["bibtex",stem], tex, tex]
            for cmd in phases:
                subprocess.run(cmd,cwd=folder,capture_output=True,check=True)
            logpath=folder/(stem+".log")
            if "Label(s) may have changed" in logpath.read_text(errors="replace"):
                subprocess.run(tex,cwd=folder,capture_output=True,check=True)
            log=logpath.read_text(errors="replace")
            (room_parent/("cleanroom_"+stem+".log")).write_text(log,encoding="utf8")
            with fitz.open(folder/(stem+".pdf")) as rebuilt, fitz.open(PKG/target) as submitted:
                identical = len(rebuilt)==len(submitted) and all(
                    a.get_pixmap().digest==b.get_pixmap().digest for a,b in zip(rebuilt,submitted))
                report[stem]=dict(pages=len(rebuilt),identical_page_pixels=identical,
                    errors=len(re.findall(r"^!|Missing character:",log,re.M)),
                    undefined=len(re.findall(r"undefined (?:references|citations)|LaTeX Warning: (?:Reference|Citation)|multiply defined|Label\(s\) may have changed",log,re.I)),
                    underfull=len(re.findall(r"Underfull \\hbox",log)),
                    overfull=len(re.findall(r"Overfull \\[hv]box",log)))
    return report


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)
    for src, dst in [("manuscript/main.pdf", "RoughCert_one_column.pdf"),
                     ("manuscript/two_columns.pdf", "RoughCert_two_columns.pdf"),
                     ("manuscript/supplement/supplement.pdf", "RoughCert_supplementary_material.pdf")]:
        shutil.copy2(ROOT / src, PKG / dst)
    src_zip = make_zip(PKG / "RoughCert_LaTeX_source.zip", latex_sources(),
                       comment="RoughCert LaTeX source; compile one_column.tex and two_columns.tex")
    (PKG / "_README_SOURCE.txt").unlink(missing_ok=True)
    code_zip = make_zip(PKG / "RoughCert_code_and_data.zip", code_sources(),
                        comment="RoughCert engines, results and regeneration scripts")

    write_highlights()
    cover_ok = write_cover_letter()
    room = clean_room(src_zip)

    files = sorted(p for p in PKG.rglob("*") if p.is_file())
    listing = "\n".join(f"- `{p.relative_to(PKG).as_posix()}` \u2014 {p.stat().st_size/1024:.0f} KB"
                        for p in files)
    write_readme(listing, room)

    files = sorted(p for p in PKG.rglob("*") if p.is_file() and p.name not in
                   {"SHA256SUMS.txt", "SHA256_MANIFEST.json"})
    lines = [f"{sha256(p)}  {p.relative_to(PKG).as_posix()}" for p in files]
    (PKG / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest = dict(package="RoughCert submission package",
                    title=TITLE, journal=JOURNAL,
                    generated=datetime.date(2026, 11, 25).isoformat(),
                    n_files=len(files),
                    files=[dict(path=p.relative_to(PKG).as_posix(), bytes=p.stat().st_size,
                                sha256=sha256(p)) for p in files])
    (PKG / "SHA256_MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n",
                                              encoding="utf-8", newline="\n")
    summary = dict(n_files=len(files) + 2, clean_room=room, cover_letter_pdf=cover_ok,
                   highlights=[len(h) for h in HIGHLIGHTS],
                   bytes=sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file()))
    (ROOT / "audit/final_package_report.json").write_text(json.dumps(summary, indent=1),
                                                                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    ok = cover_ok and all(r["pages"] and r["identical_page_pixels"] and
                         not any(r[k] for k in ("errors","undefined","overfull","underfull"))
                         for r in room.values()) and max(map(len,HIGHLIGHTS)) <= 85
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
