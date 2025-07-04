# Makefile for AutoML Distributed Platform

.PHONY: help install dev test clean docker-build docker-run k8s-deploy

help: ## Show this help message
	@echo "AutoML Distributed Platform - Available Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

dev: ## Start development environment
	docker-compose up -d
	python main.py server --reload

test: ## Run test suite
	pytest tests/ -v --cov=. --cov-report=html

test-ml: ## Run ML pipeline tests
	pytest tests/test_ml_pipeline.py -v

test-distributed: ## Run distributed simulation tests
	pytest tests/test_distributed_sim.py -v

test-chatbot: ## Run chatbot tests
	pytest tests/test_chatbot.py -v

clean: ## Clean up generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

lint: ## Run code linting
	black --check .
	flake8 .
	mypy .

format: ## Format code
	black .
	isort .

docker-build: ## Build Docker image
	docker build -t agentml-platform:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 --env-file .env agentml-platform:latest

docker-compose-up: ## Start all services with Docker Compose
	docker-compose up -d

docker-compose-down: ## Stop all services
	docker-compose down

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f k8s/

prometheus: ## Open Prometheus UI
	open http://localhost:9090

grafana: ## Open Grafana UI
	open http://localhost:3000

mlflow: ## Open MLflow UI
	open http://localhost:5000

demo: ## Run demo scenarios
	python main.py demo

init: ## Initialize platform
	python main.py init

logs: ## View application logs
	docker-compose logs -f agentml-platform

status: ## Check service status
	docker-compose ps

backup: ## Backup data
	docker-compose exec postgres pg_dump -U agentml agentml > backup.sql

restore: ## Restore data
	docker-compose exec -T postgres psql -U agentml agentml < backup.sql

# Development workflow
dev-setup: ## Setup development environment
	cp .env.example .env
	make install
	make docker-compose-up
	sleep 10
	make init

dev-reset: ## Reset development environment
	make docker-compose-down
	docker-compose rm -f
	docker volume rm $$(docker volume ls -q | grep agentml) || true
	make dev-setup

# Production deployment
prod-deploy: ## Deploy to production
	make docker-build
	make k8s-deploy

prod-update: ## Update production deployment
	make docker-build
	kubectl rollout restart deployment/agentml-platform -n agentml

# Monitoring and debugging
health: ## Check health of all services
	curl -f http://localhost:8000/health || echo "Platform: DOWN"
	curl -f http://localhost:5000/health || echo "MLflow: DOWN"  
	curl -f http://localhost:9090/-/healthy || echo "Prometheus: DOWN"
	curl -f http://localhost:3000/api/health || echo "Grafana: DOWN"

debug: ## Start debug session
	python -m pdb main.py server

profile: ## Profile application performance
	python -m cProfile -o profile.stats main.py server

# Documentation
docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html/

docs-serve: ## Serve documentation
	cd docs/_build/html && python -m http.server 8080

# CI/CD
ci-test: ## Run CI tests
	make lint
	make test
	make docker-build

release: ## Create release
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)
	make docker-build
	docker tag agentml-platform:latest agentml-platform:$(VERSION)
