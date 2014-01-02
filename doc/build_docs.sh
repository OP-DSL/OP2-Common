#!/bin/bash
#set -e
#check/install pakages
#sudo apt-get install texlive-science  python-pygments latex-xcolor

pdflatex C++_Users_Guide.tex
pdflatex C++_Users_Guide.tex
pdflatex dev.tex
pdflatex dev.tex

pdflatex --shell-escape mpi-dev.tex
pdflatex --shell-escape mpi-dev.tex
bibtex mpi-dev
pdflatex --shell-escape mpi-dev.tex

pdflatex --shell-escape airfoil-doc.tex
bibtex airfoil-doc
pdflatex --shell-escape airfoil-doc.tex
pdflatex --shell-escape airfoil-doc.tex

rm -f *.out *.aux *.blg *.pyg.* *.log *.backup *.toc *~ *.bbl

