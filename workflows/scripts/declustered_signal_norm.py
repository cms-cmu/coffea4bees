"""Normalise a (sub)sample of declustered signal MC (DeClustered D.6, signal_check.max_chunks).

    declustered_signal_norm.py <dataset.yml> <registry.yml> <output.yml>

<dataset.yml> is mixeddata_signal_check.py's synthetic_mc_<signal> dataset: the declustered files
with the ORIGINAL sample's sumw / sumw2 / count. When only part of a sample-year was declustered,
the skim registry's total_events (events processed) is below the original count: sumw and sumw2
are then scaled by total_events / count (the powheg generator weights are ~constant), so the
yields still compare like with like. A sample-year declustered in full is left unchanged.
"""
import sys

import yaml


def main():
    dataset_yml, registry_yml, output = sys.argv[1:4]
    with open(dataset_yml) as f:
        datasets = yaml.safe_load(f)
    with open(registry_yml) as f:
        registry = yaml.safe_load(f) or {}
    for name, entry in datasets.items():
        signal = name.removeprefix("synthetic_mc_")
        for year, v in entry.items():
            pico = (v or {}).get("picoAOD") if isinstance(v, dict) else None
            if not pico or not pico.get("count"):
                continue
            processed = sum(float((registry[k] or {}).get("total_events") or 0)
                            for k in registry if k == f"{signal}_{year}" or k.startswith(f"{signal}_{year}"))
            frac = processed / float(pico["count"])
            if not 0 < frac <= 1.0 + 1e-9:
                raise SystemExit(f"{name} {year}: processed {processed} of count {pico['count']}")
            if frac < 1:
                for k in ("sumw", "sumw2"):
                    if k in pico:
                        pico[k] = float(pico[k]) * frac
                pico["declustered_fraction"] = frac
            print(f"{name} {year}: declustered {processed:.0f} / {pico['count']} events -> sumw x {frac:.5f}")
    with open(output, "w") as f:
        yaml.dump(datasets, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
