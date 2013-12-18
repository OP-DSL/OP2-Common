#!/bin/bash
pdflatex C++_Users_Guide.tex
pdflatex C++_Users_Guide.tex
pdflatex dev.tex
pdflatex dev.tex
pdflatex --shell-escape mpi-dev.tex
pdflatex --shell-escape mpi-dev.tex
pdflatex --shell-escape airfoil-doc.tex
pdflatex --shell-escape airfoil-doc.tex
rm -f *.out *.bbl *.aux *.blg *.pyg.* *.log *.backup *.toc *~

