# coffea4bees/workflows/Snakefile_MvD_2c_closure.smk
# V.2c (cmslpc, --targets all_V2c): the MvD closure -- the Phase C.4 analogue. Right after the MvD
# training (V.2), before the SvB (V.3) is trained on it: four-tag data vs the mixed-data model
# mixed x JCM x MvD + TTbar4b_from_MvD, with no SvB anywhere.
#
#   V2c_check_friend                   the MvD friend (V.2) exists on EOS
#   V2c_config                         the nominal F.1 analysis config, FvT -> MvD, JCM -> V.1's fit,
#                                      run_SvB off (mvd_analysis_config(svb=False))
#   V2c_hists (dataset x year, condor) data, mixeddata_all
#   V2c_merge -> histAll_<label>.coffea, V2c_plots, V2c_cutflow (dump; `closure.known_counts` once
#   blessed), V2c_closure_table (cutflow_closure.py --multijet mixed4b: cut x year data / Bkg)

V2C_OUT = f"{out}V2c/"
V2C = config.get('closure') or {}
V2C_LABEL = V2C.get('label', "MvD_closure")
V2C_DATASETS = ['data', MIX_NAME]
V2C_CONFIG = f"{V2C_OUT}analysis_config.yml"
V2C_HISTALL = f"{V2C_OUT}histAll_{V2C_LABEL}.coffea"
V2C_FRIEND_OK = f"{V2C_OUT}friend_checked.done"
V2C_PLOT_CONFIG = V2C.get('plot_config', "coffea4bees/plots/metadata/plotsAll_MvD_roast.yml")

rule V2c_check_friend:
    output: touch(V2C_FRIEND_OK)
    log: f"{V2C_OUT}logs/check_friend.log"
    params:
        url = MVD_FRIEND.split("@@")[0]
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {log})
        u={params.url}; host=$(echo $u | sed -E 's|^(root://[^/]+)/.*|\\1|'); path=${{u#$host}}
        xrdfs $host stat $path > /dev/null 2>&1 || {{ echo "missing $u: run the falcon step V.2 (MvD_2_train) first" | tee -a {log}; exit 1; }}
        echo "found $u" | tee -a {log}
        """

rule V2c_config:
    input:
        upstream = UPSTREAM['analysis_config'][1],
        jcm = MIXED_JCM,
        friend = V2C_FRIEND_OK
    output: V2C_CONFIG
    run:
        write_yaml(output[0], mvd_analysis_config(input.upstream, signal=False, svb=False,
                                                  extra=V2C.get('config')))

use rule analysis_processor from analysis as V2c_hists with:
    input:
        runner_script = "runner.py",
        config_file = V2C_CONFIG
    output: f"{V2C_OUT}singlefiles/histAll_{V2C_LABEL}__{{dataset}}__{{year}}.coffea"
    log: f"{V2C_OUT}logs/hists__{{dataset}}__{{year}}.log"
    wildcard_constraints:
        dataset = "|".join(V2C_DATASETS),
        year = "|".join(YEARS)
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as V2c_merge with:
    input:
        files = expand(f"{V2C_OUT}singlefiles/histAll_{V2C_LABEL}__{{dataset}}__{{year}}.coffea",
                       dataset=V2C_DATASETS, year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: V2C_HISTALL
    log: f"{V2C_OUT}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

use rule make_plots from analysis as V2c_plots with:
    input:
        coffea_file = V2C_HISTALL,
        metadata_file = V2C_PLOT_CONFIG,
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{V2C_OUT}plots_{V2C_LABEL}/plots_done.txt"
    log: f"{V2C_OUT}logs/plots.log"
    params:
        output_dir = f"{V2C_OUT}plots_{V2C_LABEL}/",
        metadata = V2C_PLOT_CONFIG,
        extra_arguments = V2C.get('plot_extra_arguments', "-s xW -f png"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

use rule check_cutflow from analysis as V2c_cutflow with:
    input:
        coffea_file = V2C_HISTALL
    output:
        validation_txt = f"{V2C_OUT}cutflow_validation_{V2C_LABEL}.txt",
        cutflow_yml = f"{V2C_OUT}cutflow_{V2C_LABEL}.yml"
    log: f"{V2C_OUT}logs/cutflow.log"
    params:
        known_flag = lambda wildcards: (f'--known-cutflow "{V2C["known_counts"]}"'
                                        if V2C.get('known_counts') and os.path.exists(V2C['known_counts'])
                                        else '--known-cutflow "none"'),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

use rule cutflow_closure_table from analysis as V2c_closure_table with:
    input:
        cutflow_yml = f"{V2C_OUT}cutflow_{V2C_LABEL}.yml",
        validation_txt = f"{V2C_OUT}cutflow_validation_{V2C_LABEL}.txt"
    output:
        html = f"{V2C_OUT}cutflow_{V2C_LABEL}.html",
        txt = f"{V2C_OUT}cutflow_{V2C_LABEL}_table.txt"
    log: f"{V2C_OUT}logs/cutflow_closure.log"
    params:
        title = f"{config.get('label', 'mvd')}_cutflow_{V2C_LABEL}",
        multijet = "mixed4b",
        ttbar = "TTbar4b_from_MvD",
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

rule all_V2c:
    input:
        V2C_HISTALL,
        f"{V2C_OUT}plots_{V2C_LABEL}/plots_done.txt",
        f"{V2C_OUT}cutflow_{V2C_LABEL}.yml",
        f"{V2C_OUT}cutflow_{V2C_LABEL}.html"

localrules: V2c_check_friend, V2c_config, V2c_merge, V2c_plots, V2c_cutflow, V2c_closure_table, all_V2c
