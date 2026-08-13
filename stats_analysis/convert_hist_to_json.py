#!/usr/bin/env python
import sys
import os

# Find the path to src/tools/convert_hist_to_json.py relative to this script
script_dir = os.path.dirname(os.path.realpath(__file__))
barista_dir = os.path.realpath(os.path.join(script_dir, "../.."))
target_script = os.path.join(barista_dir, "src/tools/convert_hist_to_json.py")

if os.path.exists(target_script):
    os.execv(sys.executable, [sys.executable, target_script] + sys.argv[1:])
else:
    print(f"Error: Target script {target_script} not found.", file=sys.stderr)
    sys.exit(1)
