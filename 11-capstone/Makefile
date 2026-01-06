.PHONY: venv install train train-dl export-dl serve serve-env health test docker-build docker-run docker-build-lambda docker-run-lambda kind-create kind-load kind-apply

# ---- Config ----
PYTHON ?= python3
VENV := .venv
VENV_PY := $(VENV)/bin/python
PIP := $(VENV_PY) -m pip

HOST ?= 0.0.0.0
# host port for FastAPI
PORT ?= 9696
# container listens on this port (Dockerfile)
CONTAINER_PORT ?= 9696
THRESHOLD ?= 0.5
MODEL_BACKEND ?= rf

# Optional: for Apple Silicon you can pass:
#   make docker-run-lambda DOCKER_RUN_PLATFORM="--platform linux/amd64"
DOCKER_RUN_PLATFORM ?=

RF_MODEL_PATH ?= ./artifacts/rf_pipeline.joblib
RF_META_PATH ?= ./artifacts/rf_meta.json
RF_THRESHOLD_PATH ?= ./artifacts/rf_threshold.json

# Optional deep-learning backend (HitNet)
ONNX_PATH ?= ./artifacts/hitnet.onnx
PREPROC_PATH ?= ./artifacts/preproc.json
HITNET_THRESHOLD_PATH ?= ./artifacts/hitnet_threshold.json

EPOCHS ?= 5
DETAILS ?= data/details.csv

# ---- Python env ----
venv:
	$(PYTHON) -m venv $(VENV) --upgrade-deps || $(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m ensurepip --upgrade || true
	$(PIP) install --upgrade pip setuptools wheel

install: venv
	$(PIP) install -r requirements.txt

# ---- Training / Export ----
train: install
	$(VENV_PY) -m src.train_rf --details $(DETAILS) --artifacts artifacts

train-dl: install
	$(VENV_PY) -m src.train_torch --details $(DETAILS) --epochs $(EPOCHS)

export-dl: install
	$(VENV_PY) -m src.export_onnx --artifacts artifacts

# ---- Serve (FastAPI) ----
serve: install
	$(VENV_PY) -m uvicorn src.serve:app --host $(HOST) --port $(PORT) --reload

serve-env: install
	MODEL_BACKEND=$(MODEL_BACKEND) \
	RF_MODEL_PATH=$(RF_MODEL_PATH) RF_META_PATH=$(RF_META_PATH) RF_THRESHOLD_PATH=$(RF_THRESHOLD_PATH) \
	ONNX_PATH=$(ONNX_PATH) PREPROC_PATH=$(PREPROC_PATH) HITNET_THRESHOLD_PATH=$(HITNET_THRESHOLD_PATH) THRESHOLD=$(THRESHOLD) \
	$(VENV_PY) -m uvicorn src.serve:app --host $(HOST) --port $(PORT) --reload

health:
	curl -s http://localhost:$(PORT)/health && echo

test:
	curl -s -X POST "http://localhost:$(PORT)/predict" \
	  -H "Content-Type: application/json" \
	  -d '{"type":"TV","season":"Spring","year":2025,"episodes":12,"source":"Manga","rating":"PG-13","status":"Upcoming","genres":["Action","Sci-Fi"],"themes":[],"demographics":["Shounen"],"studios":[]}' \
	  && echo

# ---- Docker (FastAPI) ----
docker-build:
	docker build -t anime-hit-api:latest .

docker-run:
	docker run $(DOCKER_RUN_PLATFORM) --rm -p $(PORT):$(CONTAINER_PORT) \
	  -e MODEL_BACKEND=$(MODEL_BACKEND) \
	  -e RF_MODEL_PATH=/app/artifacts/rf_pipeline.joblib \
	  -e RF_META_PATH=/app/artifacts/rf_meta.json \
	  -e RF_THRESHOLD_PATH=/app/artifacts/rf_threshold.json \
	  -e ONNX_PATH=/app/artifacts/hitnet.onnx \
	  -e PREPROC_PATH=/app/artifacts/preproc.json \
	  -e HITNET_THRESHOLD_PATH=/app/artifacts/hitnet_threshold.json \
	  -e THRESHOLD=$(THRESHOLD) \
	  anime-hit-api:latest

# ---- Serverless (AWS Lambda container) ----
docker-build-lambda:
	docker buildx build --platform linux/amd64 -f docker/Dockerfile.lambda -t anime-hit-lambda:latest .

docker-run-lambda:
	docker run $(DOCKER_RUN_PLATFORM) --rm -p 9000:8080 \
	  -e MODEL_BACKEND=$(MODEL_BACKEND) \
	  -e RF_MODEL_PATH=./artifacts/rf_pipeline.joblib \
	  -e RF_META_PATH=./artifacts/rf_meta.json \
	  -e RF_THRESHOLD_PATH=./artifacts/rf_threshold.json \
	  -e ONNX_PATH=./artifacts/hitnet.onnx \
	  -e PREPROC_PATH=./artifacts/preproc.json \
	  -e HITNET_THRESHOLD_PATH=./artifacts/hitnet_threshold.json \
	  -e THRESHOLD=$(THRESHOLD) \
	  anime-hit-lambda:latest

# ---- Kubernetes (kind) ----
kind-tools-check:
	kubectl version --client
	kind version

kind-create:
	kind create cluster

kind-load:
	docker build -t anime-hit-api:latest .
	kind load docker-image anime-hit-api:latest

kind-apply:
	kubectl apply -f k8s/deployment.yaml

kind-health:
	kubectl port-forward svc/anime-hit-api-svc 9696:9696

kind-test:
	curl -s -X POST "http://localhost:$(PORT)/predict" \
	  -H "Content-Type: application/json" \
	  -d '{"type":"TV","season":"Spring","year":2025,"episodes":12,"source":"Manga","rating":"PG-13","status":"Upcoming","genres":["Action","Adventure"],"themes":["School"],"demographics":["Shounen"]}' \
	  && echo