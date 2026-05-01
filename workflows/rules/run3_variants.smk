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

if config["mode"] == "nominal":
    config.setdefault('label', '')
    config.setdefault('histogram_config',
        "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3.yml")
    config.setdefault('classifier_config',
        "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3.yml")
    config.setdefault('eos_base',
        "root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_v2")

elif config["mode"] == "quadjet_run2":
    config.setdefault('label', '_quadjet_run2')
    config.setdefault('histogram_config',
        "coffea4bees/analysis/metadata/HH4b_run_fastTopReco_Run3_quadjet_run2.yml")
    config.setdefault('classifier_config',
        "coffea4bees/analysis/metadata/HH4b_classifier_inputs_Run3_quadjet_run2.yml")
    config.setdefault('eos_base',
        "root://cmseos.fnal.gov//store/user/jda102/HH4b_Run3_quadjet_run2")

else:
    print(f"Mode {config['mode']} Not Recognized!")
    import sys
    sys.exit(-1)
