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
APP_NAME := $(shell basename $(APP_ROOT))

# Anaconda
ANACONDA_HOME ?= $(HOME)/anaconda
CONDA_ENV ?= $(APP_NAME)
CONDA_ENVS_DIR ?= $(HOME)/.conda/envs
CONDA_ENV_PATH := $(CONDA_ENVS_DIR)/$(CONDA_ENV)
DOWNLOAD_CACHE := $(APP_ROOT)/downloads
PYTHON_VERSION := 3.6

# choose anaconda installer depending on your OS
ANACONDA_URL = https://repo.continuum.io/miniconda
OS_NAME := $(shell uname -s 2>/dev/null || echo "unknown")
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
	@echo "bump             bump version using version specified as user input"
	@echo "bump-dry         bump version using version specified as user input (dry-run)"
	@echo "clean            remove all build, test, coverage and Python artifacts"
	@echo "clean-build      remove build artifacts"
	@echo "clean-env        remove package environment"
	@echo "clean-pyc        remove Python file artifacts"
	@echo "clean-test       remove test and coverage artifacts"
	@echo "lint             check style with flake8"
	@echo "test             run tests quickly with the default Python"
	@echo "test-all         run tests on every Python version with tox"
	@echo "coverage         check code coverage quickly with the default Python"
	@echo "docs             generate Sphinx HTML documentation, including API docs"
	@echo "release          package and upload a release"
	@echo "dist             package"
	@echo "install          install the package to the active Python's site-packages"
	@echo "install-docs     install docs related components"
	@echo "update           same as 'install' but without conda packages installation"

.PHONY: bump
bump: conda_env
	$(shell bash -c 'read -p "Version: " ver; echo $$ver'); \
	source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); $(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/.bumpversion.cfg --verbose --new-version $$VERSION;

.PHONY: bump-dry
bump-dry: conda_env
	$(shell bash -c 'read -p "Version: " ver; echo $$ver'); \
	source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); $(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/.bumpversion.cfg --verbose --dry-run --new-version $$VERSION;"

.PHONY: clean
clean: clean-build clean-pyc clean-test

.PHONY: clean-build
clean-build:
	rm -fr $(CUR_DIR)/build/
	rm -fr $(CUR_DIR)/dist/
	rm -fr $(CUR_DIR)/.eggs/
	find . -type f -name '*.egg-info' -exec rm -fr {} +
	find . -type f -name '*.egg' -exec rm -f {} +

.PHONY: clean-env
clean-test:
	@-test -d $(CONDA_ENV_PATH) && "$(ANACONDA_HOME)/bin/conda" remove -n $(CONDA_ENV) --yes --all

.PHONY: clean-pyc
clean-pyc:
	find . -type f -name '*.pyc' -exec rm -f {} +
	find . -type f -name '*.pyo' -exec rm -f {} +
	find . -type f -name '*~' -exec rm -f {} +
	find . -type f -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	rm -fr $(CUR_DIR)/.tox/
	rm -f $(CUR_DIR)/.coverage
	rm -fr $(CUR_DIR)/coverage/

.PHONY: lint
lint:
	flake8 api tests

.PHONY: test
test:
	python $(CUR_DIR)/setup.py test

.PHONY: test-all
test-all:
	tox

.PHONY: coverage
coverage:
	coverage run --source src/thelper setup.py test
	coverage report -m
	coverage html -d coverage
	$(BROWSER) coverage/index.html

.PHONY: docs
docs: install-docs
	echo $(CUR_DIR)
	$(CUR_DIR)/docs/sphinx "apidoc" -o $(CUR_DIR)/docs/source $(CUR_DIR)/src
	$(MAKE) -C $(CUR_DIR)/docs clean
	$(MAKE) -C $(CUR_DIR)/docs html
	$(BROWSER) $(CUR_DIR)/docs/build/html/index.html

.PHONY: install-docs
install-docs: clean conda_env
	@-bash -c "source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); pip install -r $(CUR_DIR)/docs/requirements.txt"

.PHONY: install
install: clean conda_env
	# install packages that fail with pip using conda instead
	@-bash -c "source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); $(ANACONDA_HOME)/bin/conda install -y gdal pyproj"
	@-bash -c "source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); pip install -r $(CUR_DIR)/requirements.txt"
	# enforce pip install using cloned repo
	@-bash -c "source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); pip install $(CUR_DIR)/src/thelper --no-deps"
	$(MAKE) clean

.PHONY: update
update: clean
	@-bash -c "source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); pip install $(CUR_DIR)"

# Anaconda targets

.PHONY: anaconda
anaconda:
	@echo "Installing Anaconda ..."
	@test -d $(ANACONDA_HOME) || curl $(ANACONDA_URL)/$(FN) --silent --insecure --output "$(DOWNLOAD_CACHE)/$(FN)"
	@test -d $(ANACONDA_HOME) || bash "$(DOWNLOAD_CACHE)/$(FN)" -b -p $(ANACONDA_HOME)
	@echo "Add '$(ANACONDA_HOME)/bin' to your PATH variable in '.bashrc'."

.PHONY: conda_config
conda_config: anaconda
	@echo "Update ~/.condarc"
	@"$(ANACONDA_HOME)/bin/conda" config --add envs_dirs $(CONDA_ENVS_DIR)
	@"$(ANACONDA_HOME)/bin/conda" config --set ssl_verify true
	@"$(ANACONDA_HOME)/bin/conda" config --set use_pip true
	@"$(ANACONDA_HOME)/bin/conda" config --set channel_priority true
	@"$(ANACONDA_HOME)/bin/conda" config --set auto_update_conda false
	@"$(ANACONDA_HOME)/bin/conda" config --add channels defaults
	@"$(ANACONDA_HOME)/bin/conda" config --append channels birdhouse
	@"$(ANACONDA_HOME)/bin/conda" config --append channels conda-forge

.PHONY: conda_env
conda_env: anaconda conda_config
	@echo "Update conda environment $(CONDA_ENV) using $(ANACONDA_HOME) ..."
	@test -d $(CONDA_ENV_PATH) || "$(ANACONDA_HOME)/bin/conda" create -y -n $(CONDA_ENV) python=$(PYTHON_VERSION)
	"$(ANACONDA_HOME)/bin/conda" install -y -n $(CONDA_ENV) setuptools=$(SETUPTOOLS_VERSION)
	@-bash -c "source $(ANACONDA_HOME)/bin/activate $(CONDA_ENV); pip install --upgrade pip"
