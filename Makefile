define PROJECT_HELP_MSG
Usage:
	make help                   \tshow this message\n
	make clean                  \tremove intermediate files\n
	\n	
	make install                \tmake a virtualenv in the base directory, install requirements\n
	make paper                  \tcompile the paper's .tex source\n
endef
export PROJECT_HELP_MSG

help:
	@echo "Usage:"
	@echo ""
	@echo "make help		show this message"
	@echo "make clean		remove intermediate files and clean the directory"
	@echo "make install		make a virtualenv in the base directory and install requirements"
	@echo "make paper		compile the paper's .tex source"

install:
	@echo "Setting up virtualenv..."
	@python3 -m venv .env
	@echo "Installing requirements..."
	@source .env/bin/activate
	@pip3 install -r requirements.txt
	@echo "Done."

run:
	@python3 graph.py
	@python3 stats.py
