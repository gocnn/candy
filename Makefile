# Makefile for Go projects with pre-commit integration

# Default Go commands
GO := go
GOFLAGS := -v

# Default target: run pre-commit
.PHONY: all
all: pre-commit

# Install dependencies
.PHONY: deps
deps:
	$(GO) get -u ./...
	$(GO) mod tidy
	$(GO) mod download

# Format code
.PHONY: fmt
fmt:
	$(GO) fmt ./...
	gofmt -s -w .

# Lint code (requires golangci-lint to be installed)
.PHONY: lint
lint:
	golangci-lint run --fix

# Test the project
.PHONY: test
test:
	$(GO) test $(GOFLAGS) ./...

# Pre-commit: run all checks before commit
.PHONY: pre-commit
pre-commit: deps fmt lint test
	pre-commit run --all-files

# Help message
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make          - Run pre-commit (default)"
	@echo "  make deps     - Install and tidy dependencies"
	@echo "  make fmt      - Format Go code"
	@echo "  make lint     - Run linter (requires golangci-lint)"
	@echo "  make test     - Run all tests"
	@echo "  make pre-commit - Run all pre-commit checks (deps, fmt, lint, test)"
	@echo "  make help     - Show this help message"
