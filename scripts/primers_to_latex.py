"""Compile the model primers to real LaTeX PDFs with tectonic."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_model_primers import DOCS, OUT

HEAD = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.6cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{parskip}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{xcolor}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[R]{\footnotesize\textcolor{gray}{HHP RTOFS-correction project --- model primers, 2026-09-03}}
\fancyfoot[L]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0pt}
\begin{document}
"""


def to_latex(title: str, subtitle: str, blocks) -> str:
    out = [HEAD,
           "{\\huge\\bfseries " + title + "}\\\\[2.5mm]",
           "{\\large\\itshape " + subtitle + "}\\\\[2mm]",
           "\\hrule\\vspace{5mm}", ""]
    in_list = False
    for kind, text in blocks:
        if kind != "li" and in_list:
            out.append("\\end{itemize}")
            in_list = False
        if kind == "h1":
            out.append("\\section*{" + text + "}")
        elif kind == "h2":
            out.append("\\subsection*{" + text + "}")
        elif kind == "m":
            body = text.strip()
            out.append("\\[ " + body[1:-1] + " \\]")
        elif kind == "li":
            if not in_list:
                out.append("\\begin{itemize}")
                in_list = True
            out.append("\\item " + text)
        else:
            out.append(text)
            out.append("")
    if in_list:
        out.append("\\end{itemize}")
    out.append("\\end{document}")
    return "\n".join(out)


def main() -> None:
    tectonic = shutil.which("tectonic") or str(Path.home() / "bin" / "tectonic")
    src = OUT / "src"
    src.mkdir(exist_ok=True)
    for title, sub, blocks, fname in DOCS:
        tex = src / fname.replace(".pdf", ".tex")
        tex.write_text(to_latex(title, sub, blocks))
        r = subprocess.run([tectonic, "-o", str(OUT), str(tex)], capture_output=True, text=True)
        if r.returncode == 0:
            print("ok", OUT / fname)
        else:
            print("FAILED", fname)
            print(r.stderr[-800:])


if __name__ == "__main__":
    main()
