# Contribute


### Code structure

- mistral/ — Genkit plugin wiring (models, embedders, mapping between Genkit and Mistral)
- mistralclient/ — Low-level HTTP client for Mistral (chat, embeddings, retries, rate limiting)
- internal/ — Small helpers and test fixtures
- scripts/ — Local and CI helpers

### Running tests

```bash
go test ./...
```

### Updating CI locally

Install local tooling (GitHub Actions runner via `act`):

```bash
./scripts/install_act.sh
```

Run the workflow linter:

```bash
actionlint
```

Run a workflow locally:

```bash
./scripts/run_ci_local.sh <workflow_name>
```

— Happy hacking! 🛠️