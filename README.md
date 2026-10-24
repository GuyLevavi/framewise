# Framewise

A small video analytics service used to train backend developers. It grows one week at a time;
each week is a git tag (`week-01`, `week-02`, ...).

## Run it

```bash
uv sync                                  # install from the internal index (see uv.toml)
uv run framewise probe path/to/video.mp4 --save-every 25
```

## Check it

```bash
uv run ruff check . && uv run ruff format --check .
uv run basedpyright
uv run pytest                            # unit tests and acceptance checks
```
