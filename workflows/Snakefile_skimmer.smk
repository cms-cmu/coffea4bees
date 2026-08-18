config.setdefault('output_path', "output/ttHbb/")
config.setdefault('dataset_location', "coffea4bees/metadata/datasets/archive/Run2_2024_v2/")
config.setdefault('config', "coffea4bees/skimmer/metadata/HH4b.yml")
config.setdefault('dataset', ['TTbb_Hadronic']) #GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00'])
config.setdefault('year', ['UL18'] ) #2022_EE', '2022_preEE', '2023_preBPix', '2023_BPix'])

# Valid (year, era) pairs for data PFNano in Run 3
year_era_pairs = [
    ("2022_preEE", "B"), ("2022_preEE", "C"), ("2022_preEE", "D"),
    ("2022_EE", "E"), ("2022_EE", "F"), ("2022_EE", "G"),
    ("2023_preBPix", "C01"), ("2023_preBPix", "C02"), ("2023_preBPix", "C11"), ("2023_preBPix", "C12"),
]

wildcard_constraints:
    era="[A-H]|C01|C02|C11|C12",
    # year="2022_preEE|2022_EE|2023_preBPix|2023_BPix"


rule all:
    input:
        expand( f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml", dataset=config['dataset'], year=config['year']),
        # expand( f"{config['output_path']}skimmer/picoaod_data__{{year}}_{{era}}.yml", year=[year for year, era in year_era_pairs], era=[era for year, era in year_era_pairs] )

rule skimms_mc:
    input:
        config_file = lambda wildcards: config['config'],
        processor_script = "coffea4bees/skimmer/processor/skimmer_4b.py"
    output: f"{config['output_path']}skimmer/picoaod_dataset_{{dataset}}__{{year}}.yml"
    params:
        output_path = config['output_path'],
        dataset_location = config['dataset_location'],
    log: f"{config['output_path']}logs/skimmer_dataset_{{dataset}}__{{year}}.log"
    shell:
        """
        ./run_container python runner.py -c {input.config_file} \
            --processor {input.processor_script} \
            -s \
            --output "picoaod_dataset_{wildcards.dataset}__{wildcards.year}.yml" \
            --output-path "{params.output_path}/skimmer/" \
            --datasets "{wildcards.dataset}" \
            --years "{wildcards.year}" \
            --metadata "{params.dataset_location}" \
            --condor > {log} 2>&1
        """        

# rule skimms_data:
#     input:
#         config_file = lambda wildcards: config['config'],
#         processor_script = "coffea4bees/skimmer/processor/skimmer_4b.py"
#     output: f"{config['output_path']}skimmer/picopfnano_data__{{year}}_{{era}}.yml"
#     params:
#         output_path = config['output_path'],
#         dataset_location = config['dataset_location'],
#     log: f"{config['output_path']}logs/skimmer_data__{{year}}_{{era}}.log"
#     shell:
#         """
#         ./run_container python runner.py {input.config_file} \
#             --processor {input.processor_script} \
#             -s --eras {wildcards.era} \
#             --output "picopfnano_data__{wildcards.year}_{wildcards.era}.yml" \
#             --output-path "{params.output_path}/skimmer/" \
#             --datasets "data" \
#             --years "{wildcards.year}" \
#             --metadata "{params.dataset_location}" \
#             --condor > {log} 2>&1
#         """        
