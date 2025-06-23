.PHONY: install test lint format clean run setup

# Default target when running just 'make'
help:
	@echo "Available targets:"
	@echo "  install     - Install project dependencies"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linters"
	@echo "  format     - Format code"
	@echo "  run        - Run the application"
	@echo "  clean      - Clean temporary files"
	@echo "  setup      - Set up development environment"

# Install project dependencies
install:
	poetry install

# Run tests
test:
	poetry run pytest tests/ -v

# Run linters
lint:
	poetry run flake8 src/
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/

# Format code
format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Run the application
run:
	poetry run python -m zoomcam

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	rm -rf .coverage htmlcov/

# Set up development environment
setup: install
	poetry run pre-commit install

# Run with docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f
