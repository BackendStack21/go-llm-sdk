GO ?= go

.PHONY: fmt vet lint test test-race quality clean

fmt:
	$(GO) fmt ./...

vet:
	$(GO) vet ./...

lint:
	golangci-lint run ./...

test:
	$(GO) test ./... -count=1 -timeout 120s

test-race:
	$(GO) test ./... -race -count=1 -timeout 120s

quality: fmt vet test

clean:
	$(GO) clean -testcache
