help:
	@echo "Usage:"
	@echo ""
	@echo "make help		show this message"
	@echo "make clean		remove intermediate files and clean the directory"
	@echo "make generate		generates the user.graph from data/users.csv"
	@echo "make regenerate		deletes user.graph and generates it again"
	@echo "make install		make a virtualenv in the base directory and install requirements"
	@echo "make paper		compile the paper's .tex source"

install:
	@echo "Setting up virtualenv..."
	@python3 -m venv .env
	@echo "Installing requirements..."
	@. .env/bin/activate
	@pip3 install -r requirements.txt
	@echo "Done."

clean:
	@echo "Cleaning the directory..."
	@rm -f *.log paper/*.log
	@rm -f *.aux paper/*.aux
	@rm -f *.aux paper/*.toc

generate:
	@python3 graph.py

regenerate:
	@rm -f user.graph
	@rm -f undirected.graph
	| generate

run:
	@python3 stats.py

paper:
	@pdflatex paper/main.tex
