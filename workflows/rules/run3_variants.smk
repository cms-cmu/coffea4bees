# Shared mode → config dispatch for Run3 workflows.
#
# Sets fields that are identical across:
#   - Snakefile_Run3.smk
#   - Snakefile_Run3MvD.smk
#   - Snakefile_classifier_inputs_Run3.smk
#
# Per-Snakefile fields (output_path, jcm_install_path, datasets,
# classifier_inputs_install_path) stay in each consumer because they
# differ per Snakefile (e.g. nominal Run3.smk uses a legacy committed
# classifier_inputs_Run3.json while Run3MvD.smk produces classifier_inputs_MvD_Run3.json).

config.setdefault('mode', 'nominal')

# Common across all Run3 modes
config.setdefault('analysis_container',
    "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset_location',
    "coffea4bees/metadata/datasets_HH4b_Run3/")

# Whether the analysis_processor rule submits to HTCondor. Auto-detects by
# probing for condor_submit on PATH: present on LPC, absent on falcon. Override
# with --config run_on_condor=True/False. Snakemake --config values arrive as
# strings; coerce to bool here.
import shutil
_roc = config.setdefault('run_on_condor', shutil.which("condor_submit") is not None)
config['run_on_condor'] = str(_roc).lower() not in ('false', '0', 'no')

# Mixed-data dataset name and derived suffix used to keep per-rank EOS
# outputs (trained MvD model, MvD/FvT friend trees) distinct. Empty for the
# legacy 'mixeddata_all' so historical eos_base values are preserved.
config.setdefault('dataset_name', 'mixeddata_all')
_dsn_tag = "" if config['dataset_name'] == "mixeddata_all" \
           else config['dataset_name'].removeprefix("mixeddata_all_") or config['dataset_name']
_eos_suffix = f"_{_dsn_tag}" if _dsn_tag else ""

if config["mode"] == "nominal":
    config.setdefault('label', '')
    config.setdefault('histogram_config',
        "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml")
    config.setdefault('classifier_config',
        "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3.yml")
    config.setdefault('eos_base',
        f"root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_v2{_eos_suffix}")

elif config["mode"] == "quadjet_run2":
    config.setdefault('label', '_quadjet_run2')
    config.setdefault('histogram_config',
        "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml")
    config.setdefault('classifier_config',
        "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3_quadjet_run2.yml")
    config.setdefault('eos_base',
        f"root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_quadjet_run2{_eos_suffix}")

else:
    print(f"Mode {config['mode']} Not Recognized!")
    import sys
    sys.exit(-1)
