from pathlib import Path

import yaml

with open("globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, IMAGE_ROOT) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        # data["KV_DIR"],
        data["IMAGE_ROOT"]
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
COV_PATH = data["COV_PATH"]
COV_DATA_PATH = data["COV_DATA_PATH"]
EVAL_PATH = data["EVAL_PATH"]