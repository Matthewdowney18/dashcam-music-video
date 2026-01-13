EXPORT_BASE=$(HOME)/chatgpt_uploads/dashcam_project
EXPORT_DIR=$(EXPORT_BASE)/export
EXPORT_ZIP=$(EXPORT_BASE)/dashcam_project_upload.zip

export:
	python sync_export.py --repo ../../..  --export $(EXPORT_DIR) --zip $(EXPORT_ZIP)

tree:
	python sync_export.py --repo ../../.. --export $(EXPORT_DIR)

.PHONY: export tree