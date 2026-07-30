PROJECT_NAME=image-upscale
VERSION=0.5.0
IMAGE_NAME=$(PROJECT_NAME):$(VERSION)

.PHONY: build run train test lint

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -it --gpus all \
	-v $(shell pwd):/workdir \
	--name $(PROJECT_NAME) \
	$(IMAGE_NAME) bash

train:
	python -m scripts.train

test:
	python -m pytest

lint:
	flake8 src scripts tests
