# Tutor guide

This repo is the **reference implementation**. Keep it away from the trainee; they get a
starter repo built from it. Each week is a tag (`week-01` … `week-10`, with `week-08` and
`week-09` on one commit), so you can diff what a week adds: `git diff week-03 week-04`.

## Setting up the trainee

1. Build the starter and push it to the trainee's GitLab group:

   ```bash
   uv run python scripts/make_starter.py ../framewise-starter
   cd ../framewise-starter && git init -b main && git add -A && git commit -m "Starter"
   ```

   It contains the checks, configs, `pyproject.toml` with every dependency, local pre-commit hooks,
   the merge request template and empty packages. It has no reference code, unit tests, CI file,
   Containerfile, chart or migrations: writing those is the work.
2. In GitLab, protect `main`: merge requests only, pipeline must pass, you as required approver.
3. Stage the models (README: Models) and make Postgres reachable for the checks.
4. The trainee writes `.gitlab-ci.yml` in week 3. Part of week 3's "done" is that CI runs
   `pytest checks/test_week0[1-N].py` for the current week N, so the checks gate every merge
   request from then on.

`tests/test_starter.py` proves the starter holds no reference code and that its checks fail
until the trainee works.

## Each week

Run the week's checks on the trainee's branch, then review the merge request using the list
below. The checks catch broken behaviour; they cannot judge design. Approve only when both pass.

| Week | Run | Then check in review | Common mistakes |
| --- | --- | --- | --- |
| 1 | `pytest checks/test_week01.py` | `pathlib` and context managers, no bare `except`, logging rather than `print` for diagnostics, type hints on everything | Reading the whole video into memory; swallowing errors; `os.path` string handling |
| 2 | `pytest checks/test_week0[12].py` | Classes only for FrameSource, Detector and repository; transforms as functions; dependencies passed in through constructors; tests use fakes, not real files where avoidable | A subclass per threshold; `PipelineRunner` building its own detector; tests that need a real video for pure logic |
| 3 | `...test_week0[1-3].py` | Pydantic models for every body; 404 and 422 used correctly; settings from YAML, secrets from the environment; CI caches uv | Business logic inside endpoint functions; hard-coded paths; CI that installs from the internet |
| 4 | `...test_week0[1-4].py` | One short transaction per operation; `FOR UPDATE SKIP LOCKED`; stale-job reclaim with an attempt limit; indexes match the queries; logs carry `job_id` | Long work in the API process; a session shared across threads; migrations edited after being applied |
| 5 | `...test_week0[1-5].py` | Pre- and post-processing shared by both detectors and tested on hand-computed cases; batching; timeouts and bounded retries; readable failure when Triton is down | One request per frame; BGR/RGB mix-ups; no timeout, so jobs hang |
| 6 | `...test_week0[1-6].py`, then `pytest -m eval` | Evaluation runs the production runner and detector; model and data versions logged; thresholds in config | A notebook copy of the pipeline; evaluating on frames the pipeline never samples |
| 7 | `...test_week0[1-7].py`; build with `FRAMEWISE_BUILD_IMAGE=1 FRAMEWISE_BASE_IMAGE=<quay base>` | Build tools only in the build stage; the image runs as `--user 1000620000:0`; nothing secret in any layer | `COPY . .` (pulls in `.venv` and configs); running as root; writing to directories not owned by group 0 |
| 8 | `...test_week0[1-8].py`, then live in the cluster | `oc delete pod` recovers; readiness fails when the app is unhealthy; one Splunk search on a `job_id` shows the job end to end | A liveness probe that checks the database; credentials in a ConfigMap |
| 9 | `...test_week0[1-9].py`, then live | A merge reaches nonprod with no manual step; `git revert` rolls back; devops reviewed the chart | A CI commit without `[skip ci]` (endless pipelines); `kubectl apply` by hand |
| 10 | `pytest checks` | Events SQL reviewed line by line; API key compared in constant time; limits enforced while streaming | Reading the whole upload before checking its size; a key in the repo |

## Tested here, and not

Everything was run on a machine with Python 3.13, a real PostgreSQL 16 and the official
YOLO26n ONNX model (exported with Ultralytics 8.4.188): ruff, basedpyright and `pytest` (unit tests plus checks) all pass at every
tag. These parts could **not** be run there; treat them as drafts to verify in your environment:

- **Container image:** no base image was reachable, so it was never built. The steps were
  simulated on the host instead: an offline install of a wheel, and running the API and Alembic as
  UID 1000620000 in group 0.
- **Helm chart:** `helm` was not available. The values were validated against the settings
  models, and the templates were never rendered.
- **Triton:** tested against `checks/fake_triton.py`, which speaks Triton's HTTP protocol with
  binary tensors and runs the same model, not against real Triton. `config.pbtxt` parses with
  Triton's own protobuf schema.
- **GitLab CI, Argo CD, Splunk, OpenShift:** the files are valid YAML; none of it was run.
- **Label Studio export shape:** the local label source assumes video TimelineLabels with
  1-based frame ranges. Check one real export from your Label Studio before trusting the scores.
