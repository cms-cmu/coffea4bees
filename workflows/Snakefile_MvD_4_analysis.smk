# coffea4bees/workflows/Snakefile_MvD_4_analysis.smk
# V.4 (cmslpc): the analysis with the mixed-data background model, through stat-only limits --
# nominal Phase F with mixeddata_all x JCM x MvD in place of 3b data x JCM x FvT.
#
#   V4_check_friends                   the MvD (V.2) and SvB_MvD (V.3) friends exist on EOS
#   V4_config / V4_config_signal       the nominal Phase F.1 analysis config with the FvT swapped for
#                                      the MvD, the JCM for V.1's tight mixed-data fit, and SvB_MA for
#                                      V.3's SvB_MvD. Signal: MvD off (the MvD friend covers data +
#                                      mixed data only; the processor reads event.MvD whenever
#                                      apply_MvD_weight is on)
#   V4_hists (dataset x year, condor)  data, mixeddata_all (-> the multijet model and TTbar4b_from_MvD),
#                                      the ggF signal
#   V4_merge -> histAll_<label>.coffea, V4_plots, V4_cutflow
#   stat-only limits                   Snakefile_PhaseF_2_stats.smk as is: multijet = mixeddata_all
#                                      four-tag, tt = TTbar4b_from_MvD, data_obs = data

V4_OUT = f"{out}V4/"
V4 = config.get('analysis') or {}
V4_LABEL = V4.get('label', "MvD")
V4_SIGNAL = list(V4.get('signal_datasets', ["GluGlutoHHto4B_kl-1p00_kt-1p00_c2-0p00"]))
V4_DATASETS = ['data', MIX_NAME] + V4_SIGNAL
V4_CONFIG = f"{V4_OUT}analysis_config.yml"
V4_CONFIG_SIGNAL = f"{V4_OUT}analysis_config_signal.yml"
V4_HISTALL = f"{V4_OUT}histAll_{V4_LABEL}.coffea"
V4_FRIENDS_OK = f"{V4_OUT}friends_checked.done"

rule V4_check_friends:
    output: touch(V4_FRIENDS_OK)
    log: f"{V4_OUT}logs/check_friends.log"
    params:
        urls = " ".join(u.split("@@")[0] for u in (MVD_FRIEND, SVB_FRIEND))
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        mkdir -p $(dirname {log})
        for u in {params.urls}; do
            host=$(echo $u | sed -E 's|^(root://[^/]+)/.*|\\1|'); path=${{u#$host}}
            xrdfs $host stat $path > /dev/null 2>&1 || {{ echo "missing $u: run the falcon steps (V.2 MvD, V.3 SvB) first" | tee -a {log}; exit 1; }}
            echo "found $u" | tee -a {log}
        done
        """

rule V4_config:
    input:
        upstream = UPSTREAM['analysis_config'][1],
        jcm = MIXED_JCM,
        friends = V4_FRIENDS_OK
    output: V4_CONFIG
    run:
        write_yaml(output[0], mvd_analysis_config(input.upstream, signal=False, extra=V4.get('config')))

rule V4_config_signal:
    input:
        upstream = UPSTREAM['analysis_config'][1],
        jcm = MIXED_JCM,
        friends = V4_FRIENDS_OK
    output: V4_CONFIG_SIGNAL
    run:
        write_yaml(output[0], mvd_analysis_config(input.upstream, signal=True, extra=V4.get('config')))

use rule analysis_processor from analysis as V4_hists with:
    input:
        runner_script = "runner.py",
        config_file = lambda wildcards: V4_CONFIG_SIGNAL if wildcards.dataset in V4_SIGNAL else V4_CONFIG
    output: f"{V4_OUT}singlefiles/histAll_{V4_LABEL}__{{dataset}}__{{year}}.coffea"
    log: f"{V4_OUT}logs/hists__{{dataset}}__{{year}}.log"
    wildcard_constraints:
        dataset = "|".join(V4_DATASETS),
        year = "|".join(YEARS)
    params:
        datasets = lambda wildcards: wildcards.dataset,
        years = lambda wildcards: wildcards.year,
        config = lambda wildcards, input: input.config_file,
        extra_arguments = " ".join(filter(None, [TEST_FLAG, CONDOR])),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON

use rule merging_coffea_files from analysis as V4_merge with:
    input:
        files = expand(f"{V4_OUT}singlefiles/histAll_{V4_LABEL}__{{dataset}}__{{year}}.coffea",
                       dataset=V4_DATASETS, year=YEARS),
        script = "src/tools/merge_coffea_files.py"
    output: V4_HISTALL
    log: f"{V4_OUT}logs/merge.log"
    params:
        run_performance = False,
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON,
        input_files = lambda wildcards, input: " ".join(input.files)

use rule make_plots from analysis as V4_plots with:
    input:
        coffea_file = V4_HISTALL,
        metadata_file = V4.get('plot_config', "coffea4bees/plots/metadata/plotsAll_MvD_roast.yml"),
        plot_script = "coffea4bees/plots/makePlots.py"
    output: f"{V4_OUT}plots_{V4_LABEL}/plots_done.txt"
    log: f"{V4_OUT}logs/plots.log"
    params:
        output_dir = f"{V4_OUT}plots_{V4_LABEL}/",
        metadata = V4.get('plot_config', "coffea4bees/plots/metadata/plotsAll_MvD_roast.yml"),
        extra_arguments = V4.get('plot_extra_arguments', "-s xW -f png"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

use rule check_cutflow from analysis as V4_cutflow with:
    input:
        coffea_file = V4_HISTALL
    output:
        validation_txt = f"{V4_OUT}cutflow_validation_{V4_LABEL}.txt",
        cutflow_yml = f"{V4_OUT}cutflow_{V4_LABEL}.yml"
    log: f"{V4_OUT}logs/cutflow.log"
    params:
        known_flag = lambda wildcards: (f'--known-cutflow "{V4["known_counts"]}"'
                                        if V4.get('known_counts') and os.path.exists(V4['known_counts'])
                                        else '--known-cutflow "none"'),
        error_threshold = lambda wildcards: config.get("error_threshold", "0.001"),
        cutflow_list = lambda wildcards: config.get("cutflow_list", "passJetMult,passPreSel,passDiJetMass,SR,SB"),
        run_container_wrapper = WRAPPER,
        python_bin = PYTHON
    container: None

# ── Stat-only limits: nominal Phase F.2, unchanged ────────────────────────────
# Its rules read config['output_path'] / config['label'] and histAll_<label>.coffea there; point them
# at V.4. (Nothing above reads config['output_path'] after the top-level `out` was computed.)
config['output_path'] = V4_OUT
config['label'] = V4_LABEL
config.setdefault('make_combine_inputs', {})
config['make_combine_inputs'].setdefault('multijet_process', MIX_NAME)
config['make_combine_inputs'].setdefault('tt_processes', ["TTbar4b_from_MvD"])
include: "Snakefile_PhaseF_2_stats.smk"

rule all_V4:
    input:
        V4_HISTALL,
        f"{V4_OUT}plots_{V4_LABEL}/plots_done.txt",
        f"{V4_OUT}cutflow_{V4_LABEL}.yml",
        rules.all_stats.input

localrules: V4_check_friends, V4_config, V4_config_signal, V4_merge, V4_plots, V4_cutflow, all_V4
