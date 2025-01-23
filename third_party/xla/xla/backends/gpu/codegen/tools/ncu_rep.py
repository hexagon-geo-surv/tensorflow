# Copyright 2025 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Print metrics from ncu-rep file.

Usage:
  ncu_rep -i <ncu-rep-file> [metrics|kernels|value]
    [-f <format>] [-k <kernel name>]
    [-m metric1] [-m metric2]
  metrics: print all metric names
  kernels: print all kernel names
  value (default): print values of metrics as in -m
"""

from collections.abc import Sequence
import csv
import json
import logging
import shutil
import subprocess
import sys
from absl import app
from absl import flags

_INPUT_FILE = flags.DEFINE_string(
    "i", None, "Input .ncu-rep file", required=True
)
_METRICS = flags.DEFINE_multi_string(
    "m",
    [
        "gpu__time_duration.sum",
        "sm__cycles_elapsed.max",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "launch__registers_per_thread",
    ],
    "Input .ncu-rep file",
)
_FORMAT = flags.DEFINE_enum(
    "f",
    "md",
    ["md", "csv", "json", "raw"],
    "Output format: md (default), csv, or json",
)
_KERNEL = flags.DEFINE_string(
    "k",
    None,
    "kernel to print (prints first kernel if empty)",
)

ncu_bin = shutil.which("ncu")
if not ncu_bin:
  ncu_bin = "/usr/local/cuda/bin/ncu"
logging.info("ncu binary: %s", ncu_bin)


def main(argv: Sequence[str]) -> None:
  input_name = _INPUT_FILE.value
  cmd = [ncu_bin, "-i", input_name, "--csv", "--page", "raw"]
  out = subprocess.check_output(cmd, text=True).strip()
  rows = list(csv.reader(out.splitlines()))
  name_index = {}
  for i, name in enumerate(rows[0]):
    name_index[name] = i

  op = argv[1] if len(argv) > 1 else "value"
  units = rows[1]
  if op == "metrics":
    for name in rows[0]:
      print(name)
    return

  if op == "kernels":
    for row in rows[2:]:
      print(row[name_index["Kernel Name"]])
    return

  for kernel in rows[2:]:
    if _KERNEL.value and _KERNEL.value != kernel[name_index["Kernel Name"]]:
      continue
    metrics = []
    if op == "value":
      for name in _METRICS.value:
        if name not in name_index:
          raise app.UsageError(f"metric '{name}' not found in ncu-rep file")
        idx = name_index[name]
        metrics.append([name, kernel[idx], units[idx]])
    # Print.
    fmt = _FORMAT.value
    if fmt == "csv":
      writer = csv.writer(sys.stdout)
      writer.writerow(["metric", "value", "unit"])
      writer.writerows(metrics)
    elif fmt == "json":
      d = {}
      for name, value, unit in metrics:
        d[name] = {"value": value, "unit": unit}
      print(json.dumps(d))
    elif fmt == "raw":
      for _, value, unit in metrics:
        print(value, unit)
    else:
      name_width = max(len(m[0]) for m in metrics)
      value_width = max(max(len(m[1]) for m in metrics), len("value"))
      unit_width = max(max(len(m[2]) for m in metrics), len("unit"))
      print(
          f"{'Metric'.ljust(name_width)} | {'Value'.rjust(value_width)} |"
          f" {'Unit'.ljust(unit_width)}"
      )
      print(f"{'-' * name_width }-|-{'-' * value_width }-|-{'-' * unit_width }")
      for name, value, unit in metrics:
        print(
            f"{name.ljust(name_width)} | {value.rjust(value_width)} |"
            f" {unit.ljust(unit_width)}"
        )
    break  # Only print one kernel.


if __name__ == "__main__":
  app.run(main)
