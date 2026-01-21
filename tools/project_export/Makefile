EXPORT_BASE=$(HOME)/chatgpt_uploads/dashcam_project
EXPORT_DIR=$(EXPORT_BASE)/export
EXPORT_ZIP=$(EXPORT_BASE)/dashcam_project_upload.zip

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
SYNC_SCRIPT := $(MAKEFILE_DIR)/sync_export.py
REPO_DIR := $(abspath $(MAKEFILE_DIR)/../..)

export:
	python3 $(SYNC_SCRIPT) --repo $(REPO_DIR) --export $(EXPORT_DIR) --zip $(EXPORT_ZIP)

tree:
	python3 $(SYNC_SCRIPT) --repo $(REPO_DIR) --export $(EXPORT_DIR)

.PHONY: export tree