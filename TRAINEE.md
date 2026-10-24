# Trainee guide

You build Framewise, a video analytics service, one week at a time. Each week has
acceptance checks in `checks/test_weekNN.py`. They use your code only from the outside (the
command line, HTTP and the database), so your design is your own as long as you keep the
contract below.

## Every day

```bash
uv sync                                          # install; packages come from Artifactory
uv run pytest checks/test_week03.py              # this week's checks
uv run pytest checks/test_week0[1-3].py          # this week and every earlier one
uv run pytest                                    # your own tests plus all checks
uv run ruff check . && uv run ruff format --check . && uv run basedpyright
```

From week 4 the checks need Postgres. They create and drop their own database on the server in
`FRAMEWISE_TEST_DATABASE_URL` (default: `postgres:postgres@localhost:5432`). From week 5 they need
`models/yolo26n.onnx`; ask your tutor where it is staged. A check that cannot run is
**skipped with a reason**; read the reasons (`-ra` prints them), never ignore them.

## Rules

1. **Every change goes through a merge request.** It needs a green pipeline: ruff, basedpyright,
   your tests, and the checks up to the current week.
2. **No new dependency without asking.** Everything you need is already in `pyproject.toml`. A new
   package must be mirrored into Artifactory first; ask your tutor, don't work around it.
3. **No secrets in Git, configs or images.** Passwords, tokens and keys come from environment
   variables (Secrets in the cluster). If you commit one by mistake, tell your tutor at once:
   rewriting history is not enough, the secret must be rotated.
4. **Design rules.** A class only where there is state or an interface with swappable
   implementations (frame source, detector, repository). Transforms such as pre-processing,
   IoU and metrics are plain functions. A variant of an existing algorithm is a YAML entry, never
   a subclass.
5. **Don't edit `checks/`.** If you think a check is wrong, open an issue and tell your tutor.
6. **Green checks are necessary, not sufficient.** Your tutor reviews design, naming, error
   handling, tests and logs. That review is where most of the learning happens.

## When you are stuck

Try for about 30 minutes: read the error from the top, reproduce it in the smallest possible
case, and search the official docs on the online PC. Then ask, bringing the exact command, the
full error and what you tried. A missing package, image or permission is a ticket for the admin,
not something to work around.

## The contract the checks rely on

| Week | You provide |
| --- | --- |
| 1 | `python -m shared.cli probe VIDEO [--save-every N --out DIR]` prints JSON with `width`, `height`, `fps`, `frame_count`, `duration_s`. A bad or missing file exits non-zero with a clear message, no traceback. |
| 2 | `python -m shared.cli analyze VIDEO [--detector NAME --config PATH]` prints `{"segments": [{"start_s", "end_s"}]}`. Detectors are configured in `worker/configs/detectors.yaml`. |
| 3 | The API is `api.main:app` and reads its YAML config from `FRAMEWISE_API_CONFIG`. `POST /videos` (multipart `file`) gives 201 and `video_id`. `POST /jobs` (`video_id`, `detector`) gives 202 and `id`. `GET /jobs/{id}` returns `status` (PENDING, RUNNING, DONE, FAILED), `error` and `segments`. Unknown ids give 404, unknown detectors 422. `GET /health` gives 200. |
| 4 | `python -m worker` runs the worker (config in `FRAMEWISE_WORKER_CONFIG`). Both processes read `DATABASE_URL`. `alembic upgrade head` creates `videos`, `jobs` and `detections`, with indexes on `jobs` and `detections`. |
| 5 | Detector kinds `onnx` and `triton` in `detectors.yaml`, with the entries `people-cars-local` and `people-cars-triton`. Triton is called over HTTP with batches. `GET /videos/{id}/summary` returns `[{label, detections, first_seen_s, last_seen_s}]`. |
| 6 | `pytest -m eval` reads `FRAMEWISE_EVAL_CONFIG` (same schema as `eval/eval.yaml`), prints `model=... data=...`, and fails below the thresholds. |
| 7 | A multi-stage `Containerfile`, a runtime stage writable by group 0, and a `.containerignore` that leaves out `.git`, `.venv`, `tests`, `.env` and `configs`. |
| 8–9 | `deploy/chart/` with `values-nonprod.yaml` (`image.tag`), readiness and liveness probes, a ConfigMap, a Route, ReadWriteMany storage, `DATABASE_URL` from a Secret, and Alembic as an Argo CD PreSync hook. |
| 10 | When `FRAMEWISE_API_KEY` is set, every endpoint except `/health` needs the header `X-API-Key` (otherwise 401). A wrong file type gives 415, too large 413. The events API is `events.main:app` with `GET /videos/{id}/events`. |
