# CLI & configuration

## Command-line interface

The `zea` command is defined in `zea/__main__.py` using [tyro](https://github.com/brentyi/tyro),
with argument dataclasses in `zea/cli_args.py`. There are three primary subcommands
(`docs/source/cli.rst`):

```
zea process --dataset <path> --config <config.yaml> [options]   # batch beamform a dataset
zea app [--share] [--server-port PORT]                          # launch the Gradio visualizer
zea data <operation> [options]                                  # manipulate zea data files
```

- **`process`** (`ProcessArgs`) — run a pipeline over a whole dataset.
- **`app`** (`AppArgs`) — launch the interactive Gradio visualizer (`zea/data/app.py`).
- **`data`** (`DataArgs`) — file manipulation subcommands, each a dataclass whose fields become CLI
  args and whose `run`/dispatch method does the work (`zea/cli_args.py`): `sum`, `compound_frames`,
  `compound_transmits`, `resave`, `extract`, `summary`, `copy`. Output paths are guarded unless
  `--overwrite` is passed (`_run_data_command` in `zea/cli_args.py`). `data` subcommands run without
  requiring a device (`_NO_DEVICE_FNS` in `zea/__main__.py`).

A global `device` argument cascades into the subcommands. Dataset conversion has its own entry
point (see [data.md](data.md)):

```
python -m zea.data.convert <dataset> [options]
```

## Configuration (`zea.Config`)

A config is a YAML file loaded as `zea.Config` (`zea/config.py`) that specifies where the data
lives, the pipeline to run, the device, and any parameter overrides. `check_config(config)` fills
defaults and validates (`docs/source/config.rst`):

```python
from zea import Config
from zea.config import check_config

config = Config.from_path("configs/config_picmus_rf.yaml")
config = check_config(config)
config.pipeline.operations   # list of op names / {name, params} dicts
config.to_yaml("my_config.yaml")
```

Pipeline operations in a config are strings or `{name, params}` dicts resolved through the ops
registry (see [pipeline.md](pipeline.md)). Example configs live in `configs/` (`config_picmus_rf.yaml`,
`config_picmus_iq.yaml`, `config_camus.yaml`, `config_carotid.yaml`, `config_echonet.yaml`,
`config_echonetlvh.yaml`); `configs/README.md` documents them. Configs can be loaded from
`hf://…` paths, and a GitHub Action (`.github/workflows/sync-hf-configs.yaml`) keeps the
Hugging Face copies in sync.

## Environment variables

Runtime behavior is tuned through environment variables (full table in
`docs/source/environment.rst`). The most important:

| Variable | Purpose | Default |
| --- | --- | --- |
| `KERAS_BACKEND` | ML backend: `jax`, `tensorflow`, `torch`, `numpy` | `jax` |
| `ZEA_CACHE_DIR` | Cache dir for downloaded weights/datasets | `~/.cache/zea` |
| `ZEA_LOG_LEVEL` | Log level (`DEBUG`…`CRITICAL`) | `DEBUG` |
| `ZEA_DISABLE_CACHE` | Use a temp cache deleted on exit | `0` |
| `ZEA_DOWNLOAD_TIMEOUT` / `ZEA_NVIDIA_SMI_TIMEOUT` | Network / `nvidia-smi` timeouts | `60` / `30` |
| `ZEA_TEST_DEVICE` | Restrict tests to a device | `auto:1` |

`KERAS_BACKEND` must be set **before** importing `zea` (Keras locks the backend on first import).
Secrets/tokens are read from the environment or a git-ignored `.env` (there is an `.env.example`
with placeholders); never commit real credentials. Device selection at runtime goes through
`zea.init_device()` (`zea/internal/device.py`).
