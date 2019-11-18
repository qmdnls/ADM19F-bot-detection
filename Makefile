help:
	@echo "Usage:"
	@echo ""
	@echo "make help		show this message"
	@echo "make clean		remove intermediate files and clean the directory"
	@echo "make generate		generates the user.graph from data/users.csv"
	@echo "make regenerate		deletes user.graph and generates it again"
	@echo "make install		make a virtualenv in the base directory and install requirements"
	@echo "make paper.pdf		build the paper (recommended)"
	@echo "make pdf			compile the paper's .tex source using latexmk"
	@echo "make pdflatex		compile the paper's .tex source using pdflatex, might require multiple runs"
	@echo "make all.tar		creates a .tar ready for distribution"

install:
	@echo "Setting up virtualenv..."
	@python3 -m venv .env
	@echo "Installing requirements..."
	@. .env/bin/activate
	@pip3 install -r requirements.txt
	@echo "Done."

clean:
	@echo "Cleaning the directory..."
	@rm -f paper/*.log
	@rm -f paper/*.aux
	@rm -f paper/*.toc
	@rm -f paper/*.fls
	@rm -f paper/*.fdb_latexmk
	@rm -f paper/*.bbl
	@rm -f paper/*.blg

generate:
	@python3 graph.py

regenerate:
	@rm -f user.graph
	@rm -f undirected.graph
	| generate

run:
	@python3 stats.py

paper.pdf:
	@echo "Warning: Requires latexmk. To compile manually using pdflatex please run 'make pdflatex'."
	@echo "Compiling .tex source files..."
	@cd paper && make pdf

pdf:
	@echo "Compiling .tex source files..."
	@cd paper && make pdf

pdflatex:
	@echo "Warning: Might require multiple runs to correctly generate table of contents, bibliography etc."
	@echo "Compiling .tex source files..."
	@cd paper && make pdflatex

all.tar:
	tar --exclude='*.graph' --exclude='*.tar.gz' -czvf all.tar.gz *
