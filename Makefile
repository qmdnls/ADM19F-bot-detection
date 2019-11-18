define PROJECT_HELP_MSG
Usage:
	make help                   show this message
	make clean                  remove intermediate files (see CLEANUP)
	
	make ${VENV}                make a virtualenv in the base directory (see VENV)
	make python-reqs            install python packages in requirements.pip
	make git-config             set local git configuration
	make setup                  git init; make python-reqs git-config
endef
export PROJECT_HELP_MSG

help:
	@echo $$PROJECT_HELP_MSG | less

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
