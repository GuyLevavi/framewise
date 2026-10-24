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

## Models

The detector uses Ultralytics YOLO26n (AGPL-3.0 or an Ultralytics Enterprise licence; the team
is licensed). Model files are not in Git. Export the ONNX file once, on a connected machine, in a
separate environment (Ultralytics and PyTorch are not dependencies of this repo):

```bash
pip install ultralytics onnx onnxslim
curl -LO https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
yolo export model=yolo26n.pt format=onnx imgsz=640 opset=17 dynamic=True simplify=True
cp yolo26n.onnx models/yolo26n.onnx
cp yolo26n.onnx model_repository/yolo26n/1/model.onnx     # for Triton
```

`dynamic=True` gives a variable batch size, which Triton's batching needs. The export keeps the
standard head (84 x 8400 output, NMS done by `shared/yolo.py`). Checked with
Ultralytics 8.4.188: the `.pt` has SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`. Tests that need the model are
skipped, with a reason, when it is missing.

## Evaluation gate

`uv run pytest -m eval` scores the detector on the labeled clips in `eval/data` (see
`eval/eval.yaml`). In CI the team helper library fills that folder from DagsHub and Label Studio.
To try the gate locally, first write one demo clip: `uv run python scripts/make_demo_eval_data.py`.

## Database

Tests create and drop their own database on the server named by `FRAMEWISE_TEST_DATABASE_URL`
(default `postgresql+psycopg://postgres:postgres@localhost:5432/postgres`). Locally:

```bash
podman run -d --name pg -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
```
