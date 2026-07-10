# Makefile
# 測試和開發工具快捷指令

.PHONY: help test test-unit test-integration test-e2e test-smoke test-cov lint format clean install demo-up demo-seed demo-e2e demo-benchmark portfolio-evidence demo-capture demo-down deploy-up deploy-status deploy-logs deploy-down

DEMO_PROJECT ?= sagaforge-evidence
DEMO_REPORT_DIR ?= reports
DEMO_COMPOSE = docker compose -p $(DEMO_PROJECT) -f docker-compose.prod.yml -f docker-compose.demo.yml
DEPLOY_ENV ?= .env.deploy
DEPLOY_COMPOSE = docker compose --env-file $(DEPLOY_ENV) -f docker-compose.prod.yml -f docker-compose.demo.yml

# Default target
help:
	@echo "📋 Multi-Modal Lab Backend - Available Commands:"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-e2e      - Run end-to-end tests only"
	@echo "  test-smoke    - Run quick smoke tests"
	@echo "  test-cov      - Run tests with coverage report"
	@echo ""
	@echo "🔧 Development:"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black and isort"
	@echo "  install       - Install development dependencies"
	@echo "  clean         - Clean test artifacts and cache"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make install && make test-smoke"

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt

# Testing targets
test:
	@chmod +x scripts/test_runner.sh
	@scripts/test_runner.sh all

test-unit:
	@chmod +x scripts/test_runner.sh
	@scripts/test_runner.sh unit

test-integration:
	@chmod +x scripts/test_runner.sh
	@scripts/test_runner.sh integration

test-e2e:
	@chmod +x scripts/test_runner.sh
	@scripts/test_runner.sh e2e

test-smoke:
	@chmod +x scripts/test_runner.sh
	@scripts/test_runner.sh smoke

test-cov:
	@echo "📊 Running tests with coverage..."
	@mkdir -p coverage
	@python -m pytest tests/ --cov=api --cov=core --cov=workers --cov-report=html:coverage/html --cov-report=term --cov-report=xml:coverage/coverage.xml
	@echo "📈 Coverage report generated in coverage/ directory"

# Code quality
lint:
	@echo "🔍 Running code linting..."
	@ruff check api/ core/ schemas/ workers/ tests/
	@black --check api/ core/ schemas/ workers/ tests/
	@isort --check-only api/ core/ schemas/ workers/ tests/

format:
	@echo "✨ Formatting code..."
	@black api/ core/ schemas/ workers/ tests/
	@isort api/ core/ schemas/ workers/ tests/
	@ruff check api/ core/ schemas/ workers/ tests/ --fix

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	@rm -rf .pytest_cache/
	@rm -rf coverage/
	@rm -rf htmlcov/
	@rm -rf .coverage
	@find . -type d -name __pycache__ -delete
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup completed"

demo-up:
	$(DEMO_COMPOSE) up -d --build --wait

demo-seed:
	$(DEMO_COMPOSE) exec -T api python scripts/seed_demo.py --apply --reset

demo-e2e:
	@mkdir -p $(DEMO_REPORT_DIR)
	DEMO_API_KEY=$${API_SECRET_KEY:?API_SECRET_KEY is required} python scripts/e2e_portfolio.py --compose-project $(DEMO_PROJECT) --output $(DEMO_REPORT_DIR)/e2e.json

demo-benchmark:
	@mkdir -p $(DEMO_REPORT_DIR)
	DEMO_API_KEY=$${API_SECRET_KEY:?API_SECRET_KEY is required} python scripts/benchmark_demo.py > $(DEMO_REPORT_DIR)/benchmark.json

portfolio-evidence:
	python scripts/generate_portfolio_evidence.py --coverage $(DEMO_REPORT_DIR)/coverage.json --junit $(DEMO_REPORT_DIR)/junit.xml --benchmark $(DEMO_REPORT_DIR)/benchmark.json --e2e $(DEMO_REPORT_DIR)/e2e.json

demo-capture:
	cd frontend/react && npm run capture:demo

demo-down:
	$(DEMO_COMPOSE) down -v --remove-orphans

deploy-up:
	$(DEPLOY_COMPOSE) up -d --build --wait

deploy-status:
	$(DEPLOY_COMPOSE) ps

deploy-logs:
	$(DEPLOY_COMPOSE) logs --tail=200

deploy-down:
	$(DEPLOY_COMPOSE) down --remove-orphans
