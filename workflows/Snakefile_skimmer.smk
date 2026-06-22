config.setdefault('output_path', "output/boosted_skimms/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets_HH4b_Run3/")
config.setdefault('config', "coffea4bees/skimmer/metadata/HH4b_boosted.yml")
config.setdefault('analysis_container', "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest")
config.setdefault('dataset', ['GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00'])
config.setdefault('year', ['2022_EE', '2022_preEE', '2023_preBPix', '2023_BPix'])

# Valid (year, era) pairs for data PFNano in Run 3
year_era_pairs = [
    ("2022_preEE", "B"), ("2022_preEE", "C"), ("2022_preEE", "D"),
    ("2022_EE", "E"), ("2022_EE", "F"), ("2022_EE", "G"),
    ("2023_preBPix", "C01"), ("2023_preBPix", "C02"), ("2023_preBPix", "C11"), ("2023_preBPix", "C12"),
]

wildcard_constraints:
    era="[A-H]|C01|C02|C11|C12",
    year="2022_preEE|2022_EE|2023_preBPix|2023_BPix"


rule all:
    input:
        expand( f"{config['output_path']}skimmer/picopfnano_dataset_{{dataset}}__{{year}}.yml", dataset=config['dataset'], year=config['year']),
        expand( f"{config['output_path']}skimmer/picopfnano_data__{{year}}_{{era}}.yml", year=[year for year, era in year_era_pairs], era=[era for year, era in year_era_pairs] )

rule skimms_mc:
    output: f"{config['output_path']}skimmer/picopfnano_dataset_{{dataset}}__{{year}}.yml"
    params:
        output_path = config['output_path'],
        config = config['config'],
        dataset_location = config['dataset_location'],
    log: f"{config['output_path']}logs/skimmer_dataset_{{dataset}}__{{year}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor coffea4bees/skimmer/processor/skimmer_4b.py \
            --additional-flags '-s' \
            --output-filename "picopfnano_dataset_{wildcards.dataset}__{wildcards.year}.yml" \
            --output-subdir "skimmer" \
            --output-base {params.output_path} \
            --datasets "{wildcards.dataset}" \
            --year "{wildcards.year}" \
            --dataset-metadata "{params.dataset_location}" \
            --config {params.config} \
            --condor \
            --no-test
        """        

rule skimms_data:
    output: f"{config['output_path']}skimmer/picopfnano_data__{{year}}_{{era}}.yml"
    params:
        output_path = config['output_path'],
        config = config['config'],
        dataset_location = config['dataset_location'],
    log: f"{config['output_path']}logs/skimmer_data__{{year}}_{{era}}.log"
    shell:
        """
        ./run_container bash coffea4bees/scripts/run-analysis-processor.sh \
            --processor coffea4bees/skimmer/processor/skimmer_4b.py \
            --additional-flags '-s --era {wildcards.era}' \
            --output-filename "picopfnano_data__{wildcards.year}_{wildcards.era}.yml" \
            --output-subdir "skimmer" \
            --output-base {params.output_path} \
            --datasets "data" \
            --year "{wildcards.year}" \
            --dataset-metadata "{params.dataset_location}" \
            --config {params.config} \
            --condor \
            --no-test
        """        
