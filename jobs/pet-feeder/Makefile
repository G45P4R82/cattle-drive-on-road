
IMAGE_NAME := petfeeder
CONTAINER_NAME := petfeeder

.PHONY: help build run logs stop clean

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

build: ## Build the Docker image
	docker build -t $(IMAGE_NAME) .

run: ## Run the Docker container in background
	docker run -d --name $(CONTAINER_NAME) $(IMAGE_NAME)

logs: ## Follow container logs
	docker logs -f $(CONTAINER_NAME)

stop: ## Stop and remove the container
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

clean: stop ## Remove the container and the image
	docker rmi $(IMAGE_NAME) || true
