#!/usr/bin/env python3
from __future__ import annotations

import scripts.run_bahiart_walk_probe as base
from robocop.integrations.bahiart_full_body_v114 import full_body_sensor_state

# Replace only the recorder function. The walking policy, passive memory bridge,
# episode handling and trace schema remain otherwise identical to the V11 probe.
base.full_body_sensor_state = full_body_sensor_state

if __name__ == '__main__':
    base.main()
