# Makefile
# 測試和開發工具快捷指令

.PHONY: help test test-unit test-integration test-e2e test-smoke test-cov lint format clean install

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
