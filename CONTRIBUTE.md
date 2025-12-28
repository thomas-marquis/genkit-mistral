# Contribute

### Code structure

- mistral/ — Genkit plugin wiring (models, embedders, mapping between Genkit and Mistral)
- internal/ — Small helpers and test fixtures
- scripts/ — Local and CI helpers

### Install gomock

```bash
go install go.uber.org/mock/mockgen@latest
```

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