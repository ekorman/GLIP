.PHONY: build-docker build-wheel

DOCKER_IMAGE_NAME=ghcr.io/ekorman/glip:latest

build-docker :
	docker build . -t ${DOCKER_IMAGE_NAME}

build-wheel : build-docker
	 docker run -it -v ${PWD}/dist:/code/dist ${DOCKER_IMAGE_NAME} python3.8 setup.py bdist_wheel


