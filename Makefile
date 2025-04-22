PROJECT_NAME=image-upscale
VERSION=0.0.0

IMAGE_NAME=${PROJECT_NAME}:${VERSION}
CONTAINER_NAME=${PROJECT_NAME}

build-mac:
	docker build -t ${IMAGE_NAME}-mac -f Dockerfile_mac .

build-cuda:
	docker build -t ${IMAGE_NAME}-cuda -f Dockerfile_cuda .

run-mac:
	docker run --rm -it \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(HOME)/.Xauthority:/root/.Xauthority:rw \
	-v ${shell pwd}:/workdir/ \
	--name ${CONTAINER_NAME}-mac \
	${IMAGE_NAME}-mac \
	bash

run-cuda:
	docker run --rm -it \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(HOME)/.Xauthority:/root/.Xauthority:rw \
	-v ${shell pwd}:/workdir/ \
	--name ${CONTAINER_NAME}-cuda \
	${IMAGE_NAME}-cuda \
	bash

start-mac:
	docker container start ${CONTAINER_NAME}-mac
	docker exec -it ${CONTAINER_NAME}-mac bash

start-cuda:
	docker container start ${CONTAINER_NAME}-cuda
	docker exec -it ${CONTAINER_NAME}-cuda bash

kill:
	docker kill $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)
	docker rm $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

stop:
	docker container stop $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)