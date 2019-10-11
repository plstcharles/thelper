define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

CUR_DIR := $(abspath $(lastword $(MAKEFILE_LIST))/..)
APP_ROOT := $(CURDIR)
APP_NAME := thelper

# conda
CONDA_ENV ?= $(APP_NAME)
CONDA_HOME ?= $(HOME)/conda
CONDA_ENVS_DIR ?= $(CONDA_HOME)/envs
CONDA_ENV_PATH := $(CONDA_ENVS_DIR)/$(CONDA_ENV)
DOWNLOAD_CACHE := $(APP_ROOT)/downloads

# choose conda installer depending on your OS
CONDA_URL = https://repo.continuum.io/miniconda
OS_NAME := $(shell uname -s || echo "unknown")
ifeq "$(OS_NAME)" "Linux"
FN := Miniconda3-latest-Linux-x86_64.sh
else ifeq "$(OS_NAME)" "Darwin"
FN := Miniconda3-latest-MacOSX-x86_64.sh
else
FN := unknown
endif


.DEFAULT_GOAL := help

.PHONY: all
all: help

.PHONY: help
help:
	@echo "clean            remove all build, test, coverage and Python artifacts"
	@echo "clean-all        remove EVERYTHING (including the conda environment!)"
	@echo "clean-build      remove build artifacts"
	@echo "clean-env        remove conda environment"
	@echo "clean-pyc        remove Python file artifacts"
	@echo "clean-test       remove test and coverage artifacts"
	@echo "check            check lots of things with numerous dev tools"
	@echo "test             run pytest quickly in the installed environment"
	@echo "test-all         run all checks and tests in the installed environment"
	@echo "bump             bump version using version specified as user input"
	@echo "bump-dry         bump version using version specified as user input (dry-run)"
	@echo "bump-tag         bump version using version specified as user input, tags it and commits the change in git"
	@echo "run              executes the provided arguments using the framework CLI"
	@echo "docs             generate Sphinx HTML documentation, including API docs"
	@echo "install-dev      install dev related components inside the environment"
	@echo "install-docs     install docs related components inside the environment"
	@echo "install-geo      install geospatial components inside the environment"
	@echo "install          install the package inside a conda environment"

.PHONY: clean
clean: clean-build clean-pyc clean-test

.PHONY: clean-all
clean-all: clean clean-env

.PHONY: clean-build
clean-build:
	@echo "Cleaning up build artefacts..."
	@rm -fr $(CUR_DIR)/build/
	@rm -fr $(CUR_DIR)/dist/
	@rm -fr $(CUR_DIR)/.eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -type f -name '*.egg' -exec rm -f {} +

.PHONY: clean-env
clean-env:
	@echo "Cleaning up environment artefacts..."
	@rm -fr $(CUR_DIR)/downloads/
	@test ! -d $(CONDA_ENV_PATH) || "$(CONDA_HOME)/bin/conda" remove -n $(CONDA_ENV) --yes --all -v

.PHONY: clean-pyc
clean-pyc:
	@echo "Cleaning up cache artefacts..."
	@find . -type f -name '*.pyc' -exec rm -f {} +
	@find . -type f -name '*.pyo' -exec rm -f {} +
	@find . -type f -name '*~' -exec rm -f {} +
	@find . -type f -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@echo "Cleaning up test artefacts..."
	@rm -f $(CUR_DIR)/.coverage
	@rm -f $(CUR_DIR)/.coverage.*
	@rm -fr $(CUR_DIR)/coverage/
	@rm -fr $(CUR_DIR)/htmlcov/
	@find . -name '.pytest_cache' -exec rm -fr {} +

.PHONY: check
check: install-dev
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
		python setup.py sdist && \
		twine check dist/* && \
		check-manifest $(CUR_DIR) && \
		flake8 thelper tests setup.py && \
		isort --check-only --diff --recursive thelper tests setup.py && \
		echo 'All checks passed successfully.'"

.PHONY: test
test: install-dev
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
		pytest -vvv tests"

.PHONY: test-all
test-all: check
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
		pytest --cov --cov-report=term-missing:skip-covered -vv tests"

.PHONY: bump
bump: install-dev
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
	source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
	$(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/.bumpversion.cfg \
		--verbose --allow-dirty --no-tag --new-version $$VERSION_PART patch;')

.PHONY: bump-dry
bump-dry: install-dev
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
	source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
	$(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/.bumpversion.cfg \
		--verbose --allow-dirty --dry-run --tag --tag-name "{new_version}" --new-version $$VERSION_PART patch;')

.PHONY: bump-tag
bump-tag: install-dev
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
	source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
	$(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/.bumpversion.cfg \
		--verbose --allow-dirty --tag --tag-name "{new_version}" --new-version $$VERSION_PART patch;')

.PHONY: run
run: install
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); \
		python $(CUR_DIR)/thelper/cli.py $(ARGS)"

.PHONY: docs
docs: install-docs
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); make -C docs clean && make -C docs html"
ifndef CI
	$(BROWSER) $(CUR_DIR)/docs/build/html/index.html
endif

.PHONY: install-dev
install-dev: install
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); pip install -q -r $(CUR_DIR)/requirements-dev.txt"

.PHONY: install-docs
install-docs: install
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); pip install -q -r $(CUR_DIR)/docs/requirements.txt"

.PHONY: install-geo
install-geo: install
	@"$(CONDA_HOME)/bin/conda" env update --file conda-env-geo.yml
	@echo "Successfully updated conda environment with geospatial packages."

.PHONY: install
install: conda-env
	@bash -c "source $(CONDA_HOME)/bin/activate $(CONDA_ENV); pip install -q -e $(CUR_DIR) --no-deps"
	@echo "Framework successfully installed. To activate the conda environment, use:"
	@echo "    source $(CONDA_HOME)/bin/activate $(CONDA_ENV)"

.PHONY: uninstall
uninstall: clean clean-env

.PHONY: conda-base
conda-base:
	@test -d $(CONDA_HOME) || test -d $(DOWNLOAD_CACHE) || mkdir $(DOWNLOAD_CACHE)
	@test -d $(CONDA_HOME) || test -f "$(DOWNLOAD_CACHE)/$(FN)" || curl $(CONDA_URL)/$(FN) --insecure --output "$(DOWNLOAD_CACHE)/$(FN)"
	@test -d $(CONDA_HOME) || (bash "$(DOWNLOAD_CACHE)/$(FN)" -b -p $(CONDA_HOME) && \
		echo "Make sure to add '$(CONDA_HOME)/bin' to your PATH variable in '~/.bashrc'.")

.PHONY: conda-cfg
conda_config: conda-base
	@echo "Updating conda configuration..."
	@"$(CONDA_HOME)/bin/conda" config --set ssl_verify true
	@"$(CONDA_HOME)/bin/conda" config --set use_pip true
	@"$(CONDA_HOME)/bin/conda" config --set channel_priority true
	@"$(CONDA_HOME)/bin/conda" config --set auto_update_conda false
	@"$(CONDA_HOME)/bin/conda" config --add channels defaults

# the conda-env target's dependency on conda-cfg above was removed, will add back later if needed

.PHONY: conda-env
conda-env: conda-base
	@test -d $(CONDA_ENV_PATH) || (echo "Creating conda environment at '$(CONDA_ENV_PATH)'..." && \
		"$(CONDA_HOME)/bin/conda" env create --file $(CUR_DIR)/conda-env.yml -n $(APP_NAME))
