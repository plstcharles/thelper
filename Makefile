# Included custom configs change the value of MAKEFILE_LIST
# Extract the required reference beforehand so we can use it for help target
MAKEFILE_NAME := $(word $(words $(MAKEFILE_LIST)),$(MAKEFILE_LIST))
# Include custom config if it is available
-include Makefile.config

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

# Application
CUR_DIR := $(abspath $(lastword $(MAKEFILE_NAME))/..)
APP_ROOT := $(CURDIR)
APP_NAME := thelper

# guess OS (Linux, Darwin,...)
OS_NAME := $(shell uname -s 2>/dev/null || echo "unknown")
CPU_ARCH := $(shell uname -m 2>/dev/null || uname -p 2>/dev/null || echo "unknown")

# conda
CONDA_ENV      ?= $(APP_NAME)
CONDA_HOME     ?= $(HOME)/.conda
CONDA_ENVS_DIR ?= $(CONDA_HOME)/envs
CONDA_ENV_PATH := $(CONDA_ENVS_DIR)/$(CONDA_ENV)
CONDA_BIN      := $(CONDA_HOME)/bin/conda
CONDA_ENV_REAL_TARGET_PATH := $(realpath $(CONDA_ENV_PATH))
CONDA_ENV_REAL_ACTIVE_PATH := $(realpath ${CONDA_PREFIX})
# environment already active - use it directly
ifneq ("$(CONDA_ENV_REAL_ACTIVE_PATH)", "")
	CONDA_ENV_MODE := [using active environment]
	CONDA_ENV := $(notdir $(CONDA_ENV_REAL_ACTIVE_PATH))
	CONDA_CMD :=
endif
# environment not active but it exists - activate and use it
ifneq ($(CONDA_ENV_REAL_TARGET_PATH), "")
	CONDA_ENV := $(notdir $(CONDA_ENV_REAL_TARGET_PATH))
endif
# environment not active and not found - create, activate and use it
ifeq ("$(CONDA_ENV)", "")
	CONDA_ENV := $(APP_NAME)
endif
# update paths for environment activation
ifeq ("$(CONDA_ENV_REAL_ACTIVE_PATH)", "")
	CONDA_ENV_MODE := [will activate environment]
	CONDA_CMD := source "$(CONDA_HOME)/bin/activate" "$(CONDA_ENV)";
endif
DOWNLOAD_CACHE ?= $(APP_ROOT)/downloads
PYTHON_VERSION ?= `python -c 'import platform; print(platform.python_version())'`
PYTHON_TARGET  := $(shell which python)

# choose conda installer depending on your OS
CONDA_URL = https://repo.continuum.io/miniconda
ifeq ("$(OS_NAME)", "Linux")
FN := Miniconda3-latest-Linux-x86_64.sh
else ifeq ("$(OS_NAME)", "Darwin")
FN := Miniconda3-latest-MacOSX-x86_64.sh
else
FN := unknown
endif

.DEFAULT_GOAL := help

.PHONY: all
all: help

# Auto documented help targets & sections from comments
#	- detects lines marked by double octothorpe (#), then applies the corresponding target/section markup
#   - target comments must be defined after their dependencies (if any)
#	- section comments must have at least a double dash (-)
#
# 	Original Reference:
#		https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
# 	Formats:
#		https://misc.flogisoft.com/bash/tip_colors_and_formatting
_SECTION := \033[34m
_TARGET  := \033[36m
_NORMAL  := \033[0m
.PHONY: help
# note: use "\#\#" to escape results that would self-match in this target's search definition
help:	## print this help message (default)
	@echo "$(_SECTION)=== $(APP_NAME) help ===$(_NORMAL)"
	@echo "Please use 'make <target>' where <target> is one of:"
#	@grep -E '^[a-zA-Z_-]+:.*?\#\# .*$$' $(MAKEFILE_LIST) \
#		| awk 'BEGIN {FS = ":.*?\#\# "}; {printf "    $(_TARGET)%-24s$(_NORMAL) %s\n", $$1, $$2}'
	@grep -E '\#\#.*$$' "$(APP_ROOT)/$(MAKEFILE_NAME)" \
		| awk ' BEGIN {FS = "(:|\\-\\-\\-)+.*?\\#\\# "}; \
			/\--/ {printf "$(_SECTION)%s$(_NORMAL)\n", $$1;} \
			/:/   {printf "    $(_TARGET)%-24s$(_NORMAL) %s\n", $$1, $$2} \
		'

.PHONY: info
info:		## display make information
	@echo "Makefile configuration details:"
	@echo "  Application Name   $(APP_NAME)"
	@echo "  Application Root   $(APP_ROOT)"
	@echo "  Donwload Cache     $(DOWNLOAD_CACHE)"
	@echo "  OS Name            $(OS_NAME)"
	@echo "  CPU Architecture   $(CPU_ARCH)"
	@echo "  Conda Home         $(CONDA_HOME)"
	@echo "  Conda Prefix       $(CONDA_ENV_PATH)"
	@echo "  Conda Env Name     $(CONDA_ENV)"
	@echo "  Conda Env Path     $(CONDA_ENV_REAL_ACTIVE_PATH)"
	@echo "  Conda Binary       $(CONDA_BIN)"
	@echo "  Conda Actication   $(CONDA_ENV_MODE)"
	@echo "  Conda Command      $(CONDA_CMD)"
	@echo "  Python Version     $(PYTHON_VERSION)"
	@echo "  Python Target      $(PYTHON_TARGET)"

## --- clean targets --- ##

.PHONY: clean
clean: clean-build clean-pyc clean-test     ## remove all build, test, coverage and Python artifacts

.PHONY: clean-all
clean-all: clean clean-env   ## remove EVERYTHING (including the conda environment!)

.PHONY: clean-build
clean-build:    ## remove build artifacts
	@echo "Cleaning up build artefacts..."
	@rm -fr $(CUR_DIR)/build/
	@rm -fr $(CUR_DIR)/dist/
	@rm -fr $(CUR_DIR)/.eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -type f -name '*.egg' -exec rm -f {} +

.PHONY: clean-env
clean-env:  ## remove conda environment
	@echo "Cleaning up environment artefacts..."
	@rm -fr $(CUR_DIR)/downloads/
	@test ! -d $(CONDA_ENV_PATH) || "$(CONDA_HOME)/bin/conda" remove -n $(CONDA_ENV) --yes --all -v

.PHONY: clean-pyc
clean-pyc:  ## remove Python file artifacts
	@echo "Cleaning up cache artefacts..."
	@find . -type f -name '*.pyc' -exec rm -f {} +
	@find . -type f -name '*.pyo' -exec rm -f {} +
	@find . -type f -name '*~' -exec rm -f {} +
	@find . -type f -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test: ## remove test and coverage artifacts
	@echo "Cleaning up test artefacts..."
	@rm -f $(CUR_DIR)/.coverage
	@rm -f $(CUR_DIR)/.coverage.*
	@rm -fr $(CUR_DIR)/coverage/
	@rm -fr $(CUR_DIR)/htmlcov/
	@find . -name '.pytest_cache' -exec rm -fr {} +

## --- test targets --- ##

.PHONY: check
check: install-dev  ## check lots of things with numerous dev tools
	@bash -c "$(CONDA_CMD) \
		python setup.py sdist && \
		twine check dist/* && \
		check-manifest $(CUR_DIR) && \
		flake8 thelper tests setup.py && \
		isort --check-only --diff --recursive thelper tests setup.py && \
		echo 'All checks passed successfully.'"

.PHONY: test
test: install-dev   ## run pytest quickly in the installed environment
	@bash -c "$(CONDA_CMD) \
		pytest -m 'not geo' -vvv tests"

.PHONY: test-geo
test-geo: install-dev install-geo	## run pytest quickly for 'geo' test-cases in the installed environment
	@bash -c "$(CONDA_CMD) \
		pytest -m 'geo' -vvv tests"

.PHONY: test-cov
test-cov: install-dev	## run tests with coverage in the installed environment
	@bash -c "$(CONDA_CMD) \
		pytest --cov --cov-report=term-missing:skip-covered -vv tests"

.PHONY: test-all
test-all: check test-cov test-geo   ## run all checks and tests in the installed environment

## --- version targets --- ##

.PHONY: bump
bump: install-dev   ## bump version using version specified as user input
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
	$(CONDA_CMD) \
	$(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/setup.cfg \
		--verbose --allow-dirty --no-tag --new-version $$VERSION_PART patch;')

.PHONY: bump-dry
bump-dry: install-dev   ## bump version using version specified as user input (dry-run)
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
	$(CONDA_CMD) \
	$(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/setup.cfg \
		--verbose --allow-dirty --dry-run --tag --tag-name "{new_version}" --new-version $$VERSION_PART patch;')

.PHONY: bump-tag
bump-tag: install-dev   ## bump version using version specified as user input, tags it and commits the change in git
	$(shell bash -c 'read -p "Version: " VERSION_PART; \
	$(CONDA_CMD) \
	$(CONDA_ENV_PATH)/bin/bumpversion --config-file $(CUR_DIR)/setup.cfg \
		--verbose --allow-dirty --tag --tag-name "{new_version}" --new-version $$VERSION_PART patch;')

## --- execution targets --- ##

.PHONY: run
run: install    ## executes the provided arguments using the framework CLI
	@bash -c "$(CONDA_CMD) \
		python $(CUR_DIR)/thelper/cli.py $(ARGS)"

.PHONY: docs
docs: install-docs  ## generate Sphinx HTML documentation, including API docs
	@bash -c "$(CONDA_CMD) make -C docs clean && make -C docs html"
ifndef CI
	$(BROWSER) $(CUR_DIR)/docs/build/html/index.html
endif

## --- install targets --- ##

.PHONY: install-dev
install-dev: install    ## install dev related components inside the environment
	@bash -c "$(CONDA_CMD) pip install -q -r $(CUR_DIR)/requirements-dev.txt"

.PHONY: install-docs
install-docs: install   ## install docs related components inside the environment
	@bash -c "$(CONDA_CMD) pip install -q -r $(CUR_DIR)/docs/requirements.txt"

.PHONY: install-req
install-req:   ## install base package requirement inside the environment
	@bash -c "$(CONDA_CMD) pip install -q -r $(CUR_DIR)/requirements.txt"

.PHONY: install-geo
install-geo: install    ## install geospatial components inside the environment
	@cp "$(CUR_DIR)/conda-env-geo.yml" "$(CUR_DIR)/.conda-env-geo.yml"
	@sed -i "s/thelper/$(CONDA_ENV)/g" "$(CUR_DIR)/.conda-env-geo.yml"
	@"$(CONDA_HOME)/bin/conda" env update --file "$(CUR_DIR)/.conda-env-geo.yml"
	@-rm "$(CUR_DIR)/.conda-env-geo.yml"
	@echo "Successfully updated conda environment with geospatial packages."

.PHONY: install
install: conda-env      ## install the package inside a conda environment
	@bash -c "$(CONDA_CMD) pip install -q -e $(CUR_DIR) --no-deps"
	@echo "Framework successfully installed. To activate the conda environment, use:"
	@echo "    source $(CONDA_HOME)/bin/activate $(CONDA_ENV)"

.PHONY: uninstall       ## cleans all artifacts and conda environment (alias: clean-all)
uninstall: clean clean-env

## --- docker targets --- ##

.PHONY: docker-build-base
docker-build-base:   ## builds the base docker image of thelper from source
	docker build -t thelper:base -f Dockerfile "$(CUR_DIR)"

.PHONY: docker-build-geo
docker-build-geo: docker-build-base   ## builds the docker image of thelper with geospatial components from source
	docker build -t thelper:geo -f Dockerfile-geo "$(CUR_DIR)"

.PHONY: docker-build
docker-build: docker-build-base docker-build-geo   ## builds all docker images of thelper from source

## --- conda targets --- ##

.PHONY: conda-base
conda-base:
	@test -d $(CONDA_HOME) || test -d $(DOWNLOAD_CACHE) || mkdir $(DOWNLOAD_CACHE)
	@test -d $(CONDA_HOME) || test -f "$(DOWNLOAD_CACHE)/$(FN)" || \
	    curl $(CONDA_URL)/$(FN) --insecure --location --output "$(DOWNLOAD_CACHE)/$(FN)"
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
		cp "$(CUR_DIR)/conda-env.yml" "$(CUR_DIR)/.conda-env.yml" && \
		sed -i "s/thelper/$(CONDA_ENV)/g" "$(CUR_DIR)/.conda-env.yml" && \
		"$(CONDA_HOME)/bin/conda" env create --file $(CUR_DIR)/.conda-env.yml -n $(APP_NAME))
