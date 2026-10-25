# Framewise: your ten-week assignment

Over ten weeks you build, ship and operate one real application on the team's air-gapped stack:
Framewise, a video analytics service. It mirrors the job: take a model the researchers produce,
prove it on labeled data, serve it and keep it running. Each week adds a feature to the same
codebase and brings in one internal service. Read `TRAINEE.md` first for the rules and the exact
contract the checks expect each week.

Each week below has the same shape: what the app gains, which internal service you meet, what to
read in order (a few hand-picked tutorials; reference docs are for lookup, not reading), the task,
and a "done when" check.

## Ground rules

- **No internet on the work machine.** Read documentation on the online PC beside it. Packages
  come from Artifactory and images from Quay; a missing package, image or permission is a ticket
  for the admin, not something to work around.
- **Your own sandbox.** You have a GitLab group, a nonprod OpenShift namespace, a Postgres
  database and a Splunk index you are allowed to break.
- **Models are called, not built.** Training models and designing algorithms are out of scope;
  you serve and call them.
- **The checks are your contract.** `checks/test_weekNN.py` test the CLI, the HTTP API and the
  database from outside, so your own design can pass them. From week 3, CI runs the checks for
  every week so far on each merge request, and `main` only takes merge requests with a green
  pipeline and the team lead's approval. Green checks are necessary; the review judges design.
- **Skipped means unverified.** A check that cannot run (no model, no Podman, no Helm) is skipped
  with a reason. Read the reasons (`uv run pytest -ra`); a week is not done until its checks run.

## The project

Framewise takes recorded video files, detects people and vehicles in sampled frames with a YOLO26n
model served by Triton, and reports what appeared and when. By week 10 it has:

- **Your code:** an API (FastAPI) for uploads, jobs and results; a worker that claims jobs, decodes
  frames and sends batches to Triton; an events API that turns stored detections into episodes.
- **Shared services you call:** PostgreSQL (detections, jobs, the job queue), Triton (the model),
  Splunk (JSON logs from both pods), DagsHub and Label Studio (labeled clips for the evaluation gate).
- **Delivery path:** GitLab CI (lint, types, tests, evaluation gate) builds an image into Quay and
  commits its tag to the Helm chart's values file; Argo CD syncs the chart into your namespace.

## Plan at a glance

The app works at the end of every week; each week adds to the last one.

| Week | Framewise gains | Internal service met | Skills |
| --- | --- | --- | --- |
| 1 | CLI that probes a video and saves sample frames | GitLab, Artifactory | Python, PEP 8, type hints, uv, ruff, basedpyright |
| 2 | Analysis core with a motion detector, variants in YAML | GitLab merge requests | Design rules (classes, functions, config), NumPy, pytest |
| 3 | HTTP API: upload, submit job, get result | GitLab CI and runners | FastAPI, Pydantic, YAML config, CI pipelines |
| 4 | Detection results stored in Postgres; a separate worker | PostgreSQL, Podman | SQLAlchemy, Alembic, indexes, polling workers |
| 5 | Object detection with YOLO26n, locally and on Triton | Triton Inference Server | ONNX Runtime, Triton HTTP client, pre and post-processing |
| 6 | Evaluation gate on labeled clips | DagsHub, Label Studio | Team helper library, detection metrics, pytest markers |
| 7 | Container images for API and worker | Red Hat Quay | Podman, multi-stage builds, arbitrary UIDs |
| 8 | Running in nonprod, logs searchable | OpenShift, Splunk | Deployments, mounted ConfigMaps, probes, Splunk search |
| 9 | Release and rollback by Git commit | Argo CD (nonprod) | Helm, GitOps, migration hooks |
| 10 | Composite events API, frames, dashboard, hardening | Splunk dashboards | SQL over results, API keys, load testing |

## Week by week

The team's internal PEP 8 and type-hints how-tos are the primary style reference throughout; public docs, read on the online PC, are background.

### Week 1: Probe a video (Python, style, types)

**Gains:** `framewise probe <video>` prints duration, frame rate, resolution and frame count as JSON, and saves every Nth frame as a JPEG.

**Meets:** GitLab (clone, branch, push), Artifactory through uv.

**Read, in order:**

1. [The Python Tutorial](https://docs.python.org/3/tutorial/index.html), sections 3 to 9: the official tour for people who already program. Read data structures, modules, errors and classes closely.
2. [Python Type Checking (Guide)](https://realpython.com/python-type-checking/), Real Python: how and why to annotate. Its examples use mypy; basedpyright reads the same hints.
3. [Working on projects](https://docs.astral.sh/uv/guides/projects/), uv: create a project, add dependencies, run commands.
4. [Ruff tutorial](https://docs.astral.sh/ruff/tutorial/): lint and format settings in `pyproject.toml`.
5. [basedpyright: Getting started](https://docs.basedpyright.com/latest/getting_started/getting-started/), then [Config files](https://docs.basedpyright.com/latest/configuration/config-files/) while setting it up.
6. [Argparse Tutorial](https://docs.python.org/3/howto/argparse.html) and the Basic Logging Tutorial in the [Logging HOWTO](https://docs.python.org/3/howto/logging.html).
7. [Getting Started with Videos](https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html), OpenCV: reading frames from a file.
8. [Repository local hooks](https://pre-commit.com/#repository-local-hooks), pre-commit: the only hook style that works offline.

For lookup: [PEP 8](https://peps.python.org/pep-0008/) and the team's style how-tos.

**Must know:** comprehensions, `*args` and `**kwargs`, decorators, context managers, exceptions, `pathlib`, `argparse`, `logging`, dataclasses, type hints including `Optional` and `TypedDict`.

**Task:** build the CLI with `argparse`, `pathlib`, `logging` and OpenCV's `VideoCapture`. Configure ruff and basedpyright in `pyproject.toml`, and add local pre-commit hooks. Lay it out as one repo with `shared/` now, with `worker/` and `api/` added as they arrive.

**Done when:** the tool runs from a fresh clone with two commands and no internet; ruff, `ruff format --check` and `basedpyright` pass; the corrupt sample video gives a clear error message, not a traceback.

### Week 2: Find motion (OOP, NumPy, pytest)

**Gains:** an analysis core that samples frames, detects motion between them and reports motion segments with start and end times, exposed as `framewise analyze`.

**Meets:** merge requests and code review in GitLab.

**Read, in order:**

1. [Classes](https://docs.python.org/3/tutorial/classes.html), Python tutorial.
2. [Data Classes in Python](https://realpython.com/python-data-classes/), Real Python.
3. [Python Protocols](https://realpython.com/python-protocol/), Real Python: interfaces without inheritance, the shape of `Detector` and `JobRepository`.
4. [The Composition Over Inheritance Principle](https://python-patterns.guide/gang-of-four/composition-over-inheritance/), Brandon Rhodes: why a class per variant leads to an explosion of subclasses. This is the reasoning behind this week's design rules.
5. [Models](https://pydantic.dev/docs/validation/latest/concepts/models/), Pydantic: load `detectors.yaml` into validated objects.
6. [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html), then [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).
7. pytest: [Get started](https://docs.pytest.org/en/stable/getting-started.html), [How to use fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html), [How to parametrize](https://docs.pytest.org/en/stable/how-to/parametrize.html).

Optional, deeper: [Inheritance and Composition](https://realpython.com/inheritance-composition-python/), Real Python.

**Design** rules: use a class only where there is state or an interface with swappable implementations; write transforms (pre-processing, IoU, metrics) as plain functions; and express variants of the same algorithm as YAML config, not subclasses. Legacy code with a class per variant is the habit to avoid.

- `FrameSource` yields a timestamp and a frame array from a file, at a target rate
- `Detector` is abstract, with `detect(frame) -> list[Detection]`; `MotionDetector` thresholds the NumPy difference with the previous frame and boxes the changed pixels; `FakeDetector` serves tests
- `Job` is a dataclass with an `Enum` status: PENDING, RUNNING, DONE, FAILED
- `JobRepository` is an interface, with an in-memory implementation for now
- `PipelineRunner` receives the repository, plus factories that make a frame source for each job and a detector by name, through its constructor
- `worker/configs/detectors.yaml` lists named detector setups, such as `motion-sensitive` and `motion-coarse` with different thresholds, loaded into Pydantic models; a new threshold is a YAML entry, not a new class

**Task:** implement the design with tests. Tests generate a tiny synthetic video with NumPy, a square moving across a black background, so the expected motion is known exactly.

**Done when:** motion segments on the synthetic video are correct to within one frame; a new detector plugs in without editing `PipelineRunner`; a new variant of an existing detector needs only a YAML change; the merge request is approved.

### Week 3: Serve it (FastAPI, GitLab CI)

**Gains:** an HTTP API: `POST /videos` (upload), `POST /jobs`, `GET /jobs/{id}`, `GET /health`, with interactive docs at `/docs`.

**Meets:** GitLab CI, with internal runners and the CI job image from the registry.

**Read, in order:**

1. The FastAPI tutorial, these pages in order: [First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/), [Path Parameters](https://fastapi.tiangolo.com/tutorial/path-params/), [Request Body](https://fastapi.tiangolo.com/tutorial/body/), [Request Files](https://fastapi.tiangolo.com/tutorial/request-files/), [Handling Errors](https://fastapi.tiangolo.com/tutorial/handling-errors/), [Dependencies](https://fastapi.tiangolo.com/tutorial/dependencies/), [Testing](https://fastapi.tiangolo.com/tutorial/testing/).
2. [Bigger Applications](https://fastapi.tiangolo.com/tutorial/bigger-applications/), once the app outgrows one file.
3. [Create and run your first GitLab CI/CD pipeline](https://docs.gitlab.com/ci/quick_start/), then [Create a complex pipeline](https://docs.gitlab.com/ci/quick_start/tutorial/). They use gitlab.com; the internal instance works the same way.

For lookup: [CI/CD YAML syntax reference](https://docs.gitlab.com/ci/yaml/).

**Task:** build the endpoints with Pydantic models, settings read from `api/configs/api.yaml` and validated by Pydantic, secrets from environment variables, and `PipelineRunner` injected as a dependency. Run jobs with `BackgroundTasks` for now; that choice is deliberately naive. Write `.gitlab-ci.yml` with lint, type check and test-with-coverage stages, using the internal CI image and a uv cache.

**Experiment:** submit three videos at once, then restart the API mid-job, and write down what breaks. Week 4 fixes it.

**Done when:** `/docs` describes every endpoint; errors return JSON with correct status codes; a failing test blocks the merge request; CI runs the checks for every week so far on each merge request.

### Week 4: Store results (PostgreSQL, worker)

**Gains:** detection results persist in Postgres, one row per box per frame, and a separate worker process runs the jobs. `POST /jobs` returns 202 with a job id; `GET /jobs/{id}` reports status, and results are read from the database.

**Meets:** the internal PostgreSQL server, a local Postgres in Podman, and GitLab CI service containers.

**Read, in order:**

1. [PostgreSQL Tutorial](https://www.postgresql.org/docs/current/tutorial.html), chapters 2 and 3: tables, joins, aggregates, transactions.
2. [Use The Index, Luke](https://use-the-index-luke.com/), Markus Winand: [Anatomy of an Index](https://use-the-index-luke.com/sql/anatomy) and [The Where Clause](https://use-the-index-luke.com/sql/where-clause). The clearest explanation of indexes written for developers.
3. [SQLAlchemy Unified Tutorial](https://docs.sqlalchemy.org/en/21/tutorial/index.html): engine, transactions, then the ORM sections. Use the docs version that matches the installed SQLAlchemy.
4. [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html), through autogenerate.
5. [You Don't Need a Job Queue, Postgres Already Has SKIP LOCKED](https://www.prisma.io/blog/you-dont-need-a-job-queue-postgres-already-has-skip-locked): the claim query and crash recovery. The SQL is plain Postgres; skip the Prisma setup.
6. [Concurrency and async / await](https://fastapi.tiangolo.com/async/), FastAPI: when `def` and `async def` matter.

**Key ideas:**

- Postgres holds results: low-level detections (video, frame time, class, score, box) in one table. Raw boxes are stored once, and composite views such as events are queries over them (week 10).
- Index for the way results are read: by video and time, and by class. `EXPLAIN` shows whether a query uses the index.
- Long work never runs in a request handler. The API records a job. The worker is a polling loop (`while True`, claim, process, sleep) and claims jobs with `SELECT ... FOR UPDATE SKIP LOCKED`, so several workers never take the same job.
- A job left in RUNNING by a crashed worker is reclaimed after a timeout.
- Models and the session factory live in `shared/`, imported by both `api/` and `worker/`.
- Locally, Postgres runs in Podman. SQLite is fine for a quick experiment, but it has no `SKIP LOCKED`, so the job queue is always tested on Postgres.
- Alembic is SQLAlchemy's migration tool: each schema change is a versioned script, so laptop, CI and cluster databases always match. No manual SQL on a shared database.
- From now on, logs are JSON lines from `logging` with `job_id` on every line.

**Task:** add tables for videos, jobs and detections in an Alembic migration, with indexes on detections by video and frame time, and on job status. Write `PostgresJobRepository` for the week-2 interface and a detection writer that bulk-inserts one batch of frames at a time. Add `worker/` with the polling loop and a JSON log formatter. In CI, run the database tests against a postgres service container from Quay.

**Done when:** the week-3 experiment now passes: three jobs run while `/health` stays responsive, and a restart mid-job loses nothing. A query for one video's detections between two times uses the index. `alembic upgrade head` builds the schema from an empty database, and the swap needed no change to `PipelineRunner`.

### Week 5: See objects (YOLO26n, Triton)

**Gains:** an `OnnxDetector` that runs YOLO26n in-process on the CPU, and a `TritonDetector` that sends the same inputs to Triton. `GET /videos/{id}/summary` returns counts per class with first and last seen times.

**Meets:** Triton Inference Server (model repository, health and metrics endpoints) and ONNX Runtime.

**Read, in order:**

1. [YOLO26](https://docs.ultralytics.com/models/yolo26/), then [ONNX Export for YOLO26 Models](https://docs.ultralytics.com/integrations/onnx/), Ultralytics docs: what the model is and how its ONNX file is made. For export options, look up [Model Export](https://docs.ultralytics.com/modes/export/).
2. [Get started with ONNX Runtime in Python](https://onnxruntime.ai/docs/get-started/with-python.html).
3. [Non Maximum Suppression: Theory and Implementation](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/), LearnOpenCV: IoU and NMS step by step. The code is PyTorch; write it in NumPy.
4. [Triton Quickstart](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html), following the CPU path.
5. Triton Conceptual Guide, [Part 1: Model deployment](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_1-model_deployment) and [Part 2: Improving resource utilization](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_2-improving_resource_utilization), which covers dynamic batching.
6. [Triton client README](https://github.com/triton-inference-server/client) and its simple HTTP examples.

For lookup: [Model Configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html), while writing `config.pbtxt`. For the optional DALI step: [Part 5: Model ensembles](https://github.com/triton-inference-server/tutorials/tree/main/Conceptual_Guide/Part_5-Model_Ensembles) and the [DALI backend README](https://github.com/triton-inference-server/dali_backend).

**Key ideas:**

- The model is a black box with known inputs and outputs. YOLO26n takes a 640×640 RGB image scaled to 0..1 and returns 8,400 candidate boxes; the app prepares inputs and reads outputs, and that boundary is the job.
- The detector interface becomes `detect_batch(frames)`: a deliberate refactor of the week-2 design. The motion and fake detectors adapt, and `PipelineRunner` sends frames in batches.
- Pre-processing (centred letterbox, BGR to RGB, scale to 0..1, tensor layout) and post-processing (score threshold, undo the padding and scale, non-maximum suppression) live in one class shared by both detectors. Most bugs hide there, so test it against Ultralytics' own predict() on the same frame; a BGR/RGB mix-up still finds objects, just worse.
- Triton is called over HTTP, like the team's other services. Keep binary tensor data on, so frames are not sent as JSON numbers, and send frames in batches; one request per frame is the classic slow path.
- Triton can be down or slow: set timeouts, retry a bounded number of times, then fail the job with a clear reason.

**Task:** write the shared pre- and post-processing, then both detectors. Run Triton locally in Podman on CPU with the staged model repository, then switch to the dev Triton. Unit tests use a fake client; one integration test calls Triton. Compare throughput at batch size 1 and 8 using Triton's metrics.

**Done when:** boxes drawn on saved frames match Ultralytics' own predictions on the same frames to within a couple of pixels; the detector (motion, local ONNX or Triton) is chosen by a setting; with Triton stopped, jobs fail with a readable error instead of hanging.

**Optional, for large frames on a slow network:** move pre-processing into Triton, as the team has done before. The client sends compressed JPEG bytes instead of full-resolution frames. A DALI pipeline in Triton decodes, resizes and normalises them, and a Triton ensemble passes the result to YOLO26n. The client-side pre-processing class then shrinks to JPEG encoding. Before keeping the change, compare bytes per request and throughput with the plain setup.

### Week 6: Prove the model (DagsHub, Label Studio, evaluation gate)

**Gains:** `pytest -m eval` scores the detector on labeled clips and fails below a threshold. CI runs it as a gate before any image is built.

**Meets:** DagsHub, Label Studio, and the team's helper library for both.

**Read, in order:**

1. The team helper library's README and the team's existing evaluation tests. These are the primary reference.
2. [Classification: accuracy, recall, precision](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall), Google ML Crash Course: the metrics, for a developer meeting them for the first time.
3. Label Studio: [Video frame classification template](https://labelstud.io/templates/video_frame_classification), then [Export annotations](https://labelstud.io/guide/export) for the JSON shape the tests will read.
4. DagsHub: [Label Studio integration](https://dagshub.com/docs/integration_guide/label_studio/) and [Data versioning](https://dagshub.com/docs/use_cases/data_versioning/).
5. [Working with custom markers](https://docs.pytest.org/en/stable/example/markers.html), pytest: keep `eval` out of the fast test run.

**Key ideas:**

- This is the team's real handoff: researchers produce a model, and the developer proves it on labeled data before it serves anyone.
- Frame-level labels such as "person visible" are quick to make and enough to catch a broken model or pipeline. Box labels and IoU-based metrics come once that works.
- Evaluation runs through the same `PipelineRunner` and detector as production, so it tests the service's own pre- and post-processing, not a copy in a notebook.
- Evaluation is slow and needs DagsHub and Label Studio: give it its own pytest marker, keep it out of the fast test job, and cache downloaded clips between CI runs.
- Each run records model version, data version and scores, so a change in metrics traces back to a model or a label change.

**Task:** label three to five of the sample clips at frame level in Label Studio, using the team's label config. Write evaluation tests that fetch clips and labels through the helper library, run them through `PipelineRunner`, and report per-class precision and recall. Add an `evaluate` CI stage on the main branch, with thresholds in a config file.

**Done when:** scores are identical across reruns; a deliberately wrong setting, such as a 0.95 score threshold, fails the gate; the CI log shows model version, data version and scores.

### Week 7: Package it (Podman, Quay registry)

**Gains:** one image that runs as API or worker depending on its command, built and pushed by CI once the evaluation gate passes.

**Meets:** Podman, Red Hat Quay and the CI image builder.

**Read, in order:**

1. [Basic Setup and Use of Podman](https://github.com/containers/podman/blob/main/docs/tutorials/podman_tutorial.md).
2. [Multi-stage builds](https://docs.docker.com/build/building/multi-stage/), then [Building best practices](https://docs.docker.com/build/building/best-practices/), Docker docs. They apply to Podman and Buildah unchanged.
3. [Using uv in Docker](https://docs.astral.sh/uv/guides/integration/docker/): caching and multi-stage installs with uv. Its examples run as root; this week's OpenShift rule covers the rest.
4. [Creating images](https://docs.redhat.com/en/documentation/openshift_container_platform/4.16/html/images/creating-images), OpenShift docs, section "Support arbitrary user ids". Change the version in the URL to match the cluster.
5. [Robot accounts](https://docs.projectquay.io/use_quay.html#allow-robot-access-user-repo), Project Quay docs.

For lookup: [podman kube play](https://docs.podman.io/en/latest/markdown/podman-kube-play.1.html).

**OpenShift rule to learn now:** the cluster starts containers with an arbitrary user ID in group 0 and ignores the image's `USER`. Every directory the app writes must belong to group 0 and be group-writable (`chgrp -R 0` then `chmod -R g=u`).

**Task:** write one Dockerfile with a build argument for the service, built from the internal Python base image and installing from Artifactory, with `configs/` left out through `.dockerignore`. Use `opencv-python-headless`, which needs no GUI libraries. Start API, worker and Postgres together on the workstation with `podman kube play` and a pod YAML, the same kind of YAML OpenShift reads next week. Add a CI build stage with Buildah or Kaniko that pushes to Quay with the robot account, tagged by commit SHA, only after `evaluate` passes.

**Done when:** images build from a clean checkout with no internet; they hold no compilers, tests, configs or secrets; they run with `--user 1000620000:0`; they are smaller than single-stage builds; the pod YAML starts the whole app on the workstation.

### Week 8: Run it in the cluster (OpenShift, Splunk)

**Gains:** API and worker run in the nonprod namespace, reachable through a Route, with their logs searchable in Splunk.

**Meets:** the internal OpenShift cluster (the `oc` CLI and web console) and Splunk.

**Read, in order:**

1. [Learn Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/), modules 2 to 6: Deployments, Pods, Services, scaling, rolling updates. OpenShift is Kubernetes underneath.
2. [Foundations of OpenShift](https://developers.redhat.com/learn/openshift/foundations-openshift), Red Hat Developer: the web console and the `oc` CLI on top of Kubernetes.
3. [Configure a Pod to Use a ConfigMap](https://kubernetes.io/docs/tasks/configure-pod-container/configure-pod-configmap/) and [Configure Liveness, Readiness and Startup Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/).
4. [Store persistent data using PVCs](https://developers.redhat.com/learn/openshift/store-persistent-data-red-hat-openshift-using-pvcs), Red Hat Developer.
5. [Splunk Search Tutorial](https://help.splunk.com/en/splunk-enterprise/get-started/search-tutorial), all seven parts, on the team's Splunk instead of a trial.

**Key ideas:**

- Project, Pod, Deployment, Service and Route.
- Config files: each service's `configs/` YAML becomes a ConfigMap, mounted as a file the code reads at startup. Credentials go in a Secret, exposed as environment variables.
- Readiness may check Postgres; liveness checks only the process, or a database outage restarts every pod.
- API and worker share video files: a ReadWriteMany volume if the storage class offers one, otherwise S3-compatible object storage.
- Resource requests and limits; debugging with `oc logs`, `oc describe`, `oc rsh` and events.

**Task:** write the manifests (Deployments for API and worker, Service, Route, ConfigMap, Secret, volume claim and a migration Job) as the templates of a minimal Helm chart in `deploy/chart/`, and install it by hand with `helm upgrade --install`. In Splunk, search the index by `job_id`, filter by level, count events by type, and save a search that counts failed jobs per hour.

**Done when:** the service answers through its Route; `oc delete pod` recovers cleanly; three worker replicas process jobs in parallel with no job run twice; one Splunk search on a `job_id` shows that job's whole life, from upload to DONE.

### Week 9: Release by commit (Argo CD, Helm)

**Gains:** releasing and rolling back are Git commits; nobody runs `helm` or `oc` by hand any more.

**Meets:** the nonprod Argo CD that devops runs, and Helm.

**Read, in order:**

1. [OpenGitOps principles](https://opengitops.dev/): four principles on one page.
2. [Helm Chart Template Guide: Getting Started](https://helm.sh/docs/chart_template_guide/getting_started/) and the pages after it, through values files and template functions.
3. Argo CD: [Core Concepts](https://argo-cd.readthedocs.io/en/stable/core_concepts/), [Helm](https://argo-cd.readthedocs.io/en/stable/user-guide/helm/), [Automated Sync Policy](https://argo-cd.readthedocs.io/en/stable/user-guide/auto_sync/), and [Sync Phases and Waves](https://argo-cd.readthedocs.io/en/stable/user-guide/sync-waves/), which includes a PreSync migration Job.
4. One of devops' existing charts, as house style.

For lookup: [Chart Best Practices](https://helm.sh/docs/chart_best_practices/).

**Minimal setup:** one request to devops; everything else lives in Git.

1. The chart lives in the app repo under `deploy/chart/`, with a `values-nonprod.yaml`. For a small project, one repo is easier to follow and manage. Larger projects keep a separate Helm charts repo per project, managed by devops through Argo CD; the steps below stay the same, only the repo changes.
2. Devops adds one Argo CD Application in nonprod. Its source is that repo path and values file, its destination your nonprod namespace, and it starts with manual sync. If their Argo uses an app-of-apps or an ApplicationSet, this is one more entry there.
3. Argo needs read access to the repo, through a GitLab deploy token, and rights in the namespace, which managed nonprod namespaces usually grant already.
4. Once a manual sync works, devops turns on automated sync with self-heal.

**Key ideas:**

- Desired state lives in Git, live state in the cluster; "out of sync" and "degraded" are different problems.
- A Helm chart is templates plus a values file per environment; the image tag is just a value.
- A rollback is a `git revert`.

**Task:** bring the week-8 chart to devops' conventions, checking the output with `helm template` locally and `helm lint` in CI, and add `values-nonprod.yaml`. Write `deploy/argocd-application.yaml`, the one object to hand devops. Make the last CI stage commit the new image tag to `values-nonprod.yaml` with `[skip ci]` in the message, so the commit doesn't start another pipeline. Turn the migration Job into an Argo CD PreSync hook, so the schema changes before new pods start.

**Done when:** a merge to main reaches nonprod with no manual step; a revert rolls it back; a hand edit in the cluster is reported and healed; devops has reviewed the chart; you can explain a failed sync versus an unhealthy application.

### Week 10: Make it useful and safe (capstone)

**Gains:**

- **Composite events:** a separate events API that reads raw detections and returns events such as "person and car boxes overlap", grouped into episodes with start and end times. Event rules live in YAML: the classes, minimum overlap, minimum duration, and the gap that still counts as one episode.
- **Frames:** `GET /videos/{id}/frames/{t}` returns the frame at time t with boxes drawn
- **Dashboard:** a Splunk dashboard of jobs per hour, failure rate, and processing time per minute of video
- **Hardening:** an API key checked on every request, upload size and type limits, input validation

**Key ideas:**

- Events are SQL over stored results: a self-join of detections on the same video and frame, filtered by box overlap, then consecutive frames grouped into episodes (the "gaps and islands" pattern). The week-4 index on video and frame time is what keeps this fast.
- Each episode gets a deterministic key, a hash of video, rule name and start time, so reruns never create duplicates.
- An API key is a shared secret the client sends in a request header (for example `X-API-Key`). A FastAPI dependency compares it with the value from a Secret and returns 401 when it is missing or wrong. That is enough inside the network; per-user login through a company identity provider can come later.
- Upload limits reject files over a size and non-video types before reading them. Pydantic validation rejects bad ids and negative times with a 422 instead of a crash.

**Meets:** Splunk dashboards, and Locust from Artifactory.

**Read, in order:**

1. [Window Functions](https://www.postgresql.org/docs/current/tutorial-window.html), PostgreSQL tutorial: the tool for grouping consecutive frames into episodes.
2. [Security: first steps](https://fastapi.tiangolo.com/tutorial/security/), skimmed, then [APIKeyHeader](https://fastapi.tiangolo.com/reference/security/#fastapi.security.APIKeyHeader), FastAPI.
3. [OWASP API Security Top 10, 2023](https://api-security.owasp.org/editions/2023/en/0x11-t10/): what goes wrong in real APIs.
4. [Locust: Your first test](https://docs.locust.io/en/stable/quickstart.html).

**Task:** build the features above. Run a Locust test that submits 20 videos at once, and watch the workers, Triton's metrics and Postgres while it runs. Write the README: how to run locally, how it is deployed, where its logs are.

**Done when:** every item in the sign-off below is ticked.

## Sign-off

You demonstrate Framewise to the team lead, who ticks each item only when it works in the cluster, not just in code.

- [ ] Code is designed to the week-2 rules (classes for interfaces and state, functions for transforms, variants in YAML), fully typed, and passes ruff and basedpyright
- [ ] Unit, API and real-Postgres tests pass in CI, and a failing test blocks a merge
- [ ] Submit, poll and results work; a failed job, including one caused by Triton being down, reports a clear error
- [ ] Three worker replicas process jobs without running any job twice, and a pod restart loses no job
- [ ] YOLO26n runs locally on CPU and through Triton with batching, chosen by a setting, and its boxes match Ultralytics' own predictions
- [ ] The evaluation gate scores the model on labeled clips from DagsHub and Label Studio, and blocks a model below threshold
- [ ] Composite events come from SQL over the raw detections, with rules in YAML, and a rerun creates no duplicate episodes
- [ ] The image builds offline, runs as a random user in group 0, and holds no build tools or secrets
- [ ] Alembic builds the schema from an empty database and runs as an Argo CD PreSync Job
- [ ] The Helm chart follows devops conventions; a merge to main deploys with no manual step, and a rollback by `git revert` has been shown
- [ ] Configs come from ConfigMaps and secrets from a Secret; no credentials are in Git or in the image
- [ ] One Splunk search on a `job_id` shows the job's whole life, and the dashboard shows failure rate
- [ ] Endpoints require an API key, and oversized or wrong-type uploads are rejected
- [ ] The README covers running locally, deployment and where to find logs

## After sign-off

These build on Framewise when there is time or a real need; none is required for sign-off.

- **Live streams:** read an RTSP camera feed instead of a file; the `FrameSource` interface from week 2 should make this a new class, not a rewrite.
- **Box-level evaluation:** extend the week-6 gate from frame labels to boxes, with IoU-matched precision and recall.
- **User login:** per-user authentication through the company identity provider, once IT says what exists.
- **Metrics and alerts:** expose Prometheus metrics if the cluster's monitoring collects them.

Out of scope throughout: training models and designing algorithms.
