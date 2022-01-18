SOURCES=$(wildcard *.tex)

all: $(SOURCES:.tex=.png)

.PHONY: clean
clean:
	rm -f *.aux
	rm -f *.log
	rm -f *.pdf

%.png: %.tex
	xelatex -shell-escape $?
