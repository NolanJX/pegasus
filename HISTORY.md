# History

## uv

```bash
uv init . --python 3.10
```

## pre-commit

```bash
uv add --dev pre-commit
uv run pre-commit sample-config > .pre-commit-config.yaml
```

Update the `rev` of [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks) from `v3.2.0` to `v6.0.0`.

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Ruff

```bash
uv add --dev ruff
```

Configure lint rules per the [official guide](https://docs.astral.sh/ruff/linter/#rule-selection).

Wire up the pre-commit hook following the [integrations guide](https://docs.astral.sh/ruff/tutorial/#integrations).

## ty

```bash
uv add --dev ty
```

Wire up the pre-commit hook following the [ty-pre-commit](https://github.com/astral-sh/ty-pre-commit). Keep `rev` in sync with the dev dependency in `pyproject.toml`, and use `--isolated` so the hook does not create or update `uv.lock` or the local virtual environment.

```bash
SKIP=ty git commit -m "build: add ty with pre-commit hook"
```
