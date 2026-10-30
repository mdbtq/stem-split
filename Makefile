# PyTorch has no wheels for every new Python release yet; 3.12 is the tested one.
PYTHON_VERSION := 3.12
VENV           := .venv
PYTHON         := $(VENV)/bin/python

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} \
		/^##@ / {printf "  \033[1m%s\033[0m\n", substr($$0, 5); next} \
		/^[a-zA-Z0-9_\/.-]+:.*?## / {printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)

##@ Build

.PHONY: build/venv
build/venv: ## Create .venv and install the package in editable mode
	uv venv --allow-existing --python $(PYTHON_VERSION) $(VENV)
	uv pip install --python $(PYTHON) -e .

.PHONY: build/install
build/install: ## Install the stem-split command for this user (uv tool)
	uv tool install --force --python $(PYTHON_VERSION) .

.PHONY: build/uninstall
build/uninstall: ## Remove the installed command
	uv tool uninstall stem-split

.PHONY: build/clean
build/clean: ## Remove build artefacts
	rm -rf build dist src/*.egg-info

##@ Test

.PHONY: test/all
test/all: build/venv ## Run the test suite (no model needed)
	cd tests && ../$(PYTHON) -m unittest -v

.PHONY: test/model
test/model: build/venv ## Also run the test against the real htdemucs model
	cd tests && STEM_SPLIT_MODEL_TESTS=1 ../$(PYTHON) -m unittest -v
