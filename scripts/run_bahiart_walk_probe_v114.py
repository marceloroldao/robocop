#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path

from robocop.integrations.bahiart_full_body_v114 import full_body_sensor_state

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "run_bahiart_walk_probe.py"

spec = importlib.util.spec_from_file_location("robocop_bahiart_walk_probe_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load base walk probe from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Replace only the recorder function. The walking policy, passive memory bridge,
# episode handling and trace schema remain otherwise identical to the V11 probe.
base.full_body_sensor_state = full_body_sensor_state

if __name__ == "__main__":
    base.main()
