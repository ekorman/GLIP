.PHONY: build-docker build-wheel build-build-docker

DOCKER_IMAGE_NAME=ghcr.io/ekorman/glip

build-docker :
	docker build . -t ${DOCKER_IMAGE_NAME}

publish-docker : build-docker
	docker push ${DOCKER_IMAGE_NAME}

publish-py38-whl :
	docker build --build-arg PYTHONVERSION=38 -t glip-build-py38 -f Dockerfile.build .
	docker run -e PYPITOKEN glip-build-py38

publish-py39-whl :
	docker build --build-arg PYTHONVERSION=39 -t glip-build-py39 -f Dockerfile.build .
	docker run -e PYPITOKEN glip-build-py39