#!/usr/bin/env python
import sys
import os
import hist
import numpy as np

def hist_to_json( coffea_hist ):
    """docstring for hist_to_root"""

    yhist = {
        'edges' : coffea_hist.axes[0].edges.tolist(),
        'centers' : coffea_hist.axes[0].centers.tolist(),
        'values' : coffea_hist.values().tolist(),
        'variances' : coffea_hist.variances().tolist(),
        'underflow_value' : coffea_hist[hist.loc(-np.inf)].value,
        'underflow_variance' : coffea_hist[hist.loc(-np.inf)].variance,
        'overflow_value' : coffea_hist[hist.loc(+np.inf)].value,
        'overflow_variance' : coffea_hist[hist.loc(+np.inf)].variance,
    }
    return yhist

if __name__ == '__main__':
    # Find the path to src/tools/convert_hist_to_json.py relative to this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    barista_dir = os.path.realpath(os.path.join(script_dir, "../.."))
    target_script = os.path.join(barista_dir, "src/tools/convert_hist_to_json.py")

    if os.path.exists(target_script):
        os.execv(sys.executable, [sys.executable, target_script] + sys.argv[1:])
    else:
        print(f"Error: Target script {target_script} not found.", file=sys.stderr)
        sys.exit(1)
