help:
	@echo "Usage:"
	@echo ""
	@echo "make help                show this message"
	@echo "make clean               remove intermediate files and clean the directory"
	@echo "make pdf                 compile the paper's .tex source using latexmk (recommended)"
	@echo "make pdflatex            compile the paper's .tex source using pdflatex, might require multiple runs"

pdf:
	latexmk -pdf -use-make main.tex

pdflatex:
	pdflatex main.tex
