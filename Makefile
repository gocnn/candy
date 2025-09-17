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

# Generate code
.PHONY: gen
gen:
	$(GO) generate ./...

# Build the project
.PHONY: build
build:
	$(GO) build $(GOFLAGS) ./...

# Benchmark the project
.PHONY: bench
bench:
	$(GO) test -run=NO_TEST -bench . -benchmem -benchtime 3s ./...

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
	@echo "  make gen      - Generate code"
	@echo "  make build    - Build the project"
	@echo "  make bench    - Run benchmarks"
	@echo "  make pre-commit - Run all pre-commit checks (deps, fmt, lint, test)"
	@echo "  make help     - Show this help message"
