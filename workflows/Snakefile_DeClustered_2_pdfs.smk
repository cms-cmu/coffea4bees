# coffea4bees/workflows/Snakefile_DeClustered_2_pdfs.smk
# D.2: distill the D.1 splitting histograms into the per-era sampling PDFs the DeClusterer draws
# from (jet_clustering/make_jet_splitting_PDFs.py; until now run by hand and the output committed
# as jet_clustering/jet-splitting-PDFs-<version>/).
#
#   D2_make_pdfs (per era)            -> D2/per_era/<era>/clustering_pdfs_vs_pT_<era>.yml + sampling-test
#                                        plots (one directory per era: the tool names its test_sampling_*
#                                        plots without the era, so a shared directory keeps only the last)
#   D2_publish                        PDFs -> <PDF_BASE>/ on EOS
#
# The declustering jobs (D.3) run on condor workers and read the PDFs from EOS through fsspec
# (make_declustered_data_4b.py), so nothing is installed into the checkout.
# With inputs.pdfs set, D.3 reads another roast's PDFs and none of this runs.

D2_OUT = f"{out}D2/"
D2_ERA_DIR = f"{D2_OUT}per_era/"
D2_PDFS = [f"{D2_ERA_DIR}{y}/clustering_pdfs_vs_pT_{y}.yml" for y in YEARS]
D2_PUBLISHED = f"{D2_OUT}published.done"
D2_DONE = [] if PDF_EXTERNAL else [D2_PUBLISHED]

rule D2_make_pdfs:
    input: D1_MERGED
    output: f"{D2_ERA_DIR}{{year}}/clustering_pdfs_vs_pT_{{year}}.yml"
    log: f"{D2_OUT}logs/make_pdfs__{{year}}.log"
    wildcard_constraints:
        year = "|".join(YEARS)
    shell:
        """
        set -eo pipefail
        export MPLCONFIGDIR="/tmp/matplotlib"
        mkdir -p $MPLCONFIGDIR $(dirname {output})
        {WRAPPER} {PYTHON} coffea4bees/jet_clustering/make_jet_splitting_PDFs.py {input} \
            -o $(dirname {output})/ --years {wildcards.year} 2>&1 | tee {log}
        ls -l {output} 2>&1 | tee -a {log}
        """

rule D2_publish:
    input: D2_PDFS
    output: D2_PUBLISHED
    log: f"{D2_OUT}logs/publish.log"
    shell:
        """
        set -eo pipefail
        {EOS_PROXY}
        for f in {input}; do
            xrdcp -f -p "$f" "{PDF_BASE}/$(basename $f)" 2>&1 | tee -a {log}
            echo "published $f -> {PDF_BASE}/$(basename $f)" | tee -a {log}
        done
        date > {output}
        """

rule all_D2:
    input: D2_DONE

localrules: D2_make_pdfs, D2_publish, all_D2
