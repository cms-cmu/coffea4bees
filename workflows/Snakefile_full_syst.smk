# ==============================================================================
# coffea4bees/workflows/Snakefile_full_syst.smk
#
# Master Coordinator for Full Systematics Pipeline (Signal Systs + Stats)
# ==============================================================================
import os
from datetime import datetime

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/ttHbb_full_syst.yml"

cern_user = config.get("cern_user", os.environ.get("USER", "algomez"))
cern_path = config.get("cern_path", "www/ttHbb/Plots/")
if not cern_path.startswith("/"):
    first_letter = cern_user[0]
    cern_path = f"/eos/user/{first_letter}/{cern_user}/{cern_path}"

include: "Snakefile_full_syst_1_analysis.smk"
include: "Snakefile_full_syst_2_stats.smk"

rule final_output:
    default_target: True
    input:
        rules.all_full_syst_F_1.input,
        rules.all_full_syst_F_2.input
    params:
        output_dir = f"{datetime.now().strftime('%Y%m%d')}_{config['label']}/",
        cern_path = cern_path,
        email = lambda wildcards: config.get('email', "")
    shell:
        """
        echo "Copying full systematics results to EOS..."
        if [ -f "proxy/x509_proxy" ]; then
            export X509_USER_PROXY="$(pwd)/proxy/x509_proxy"
        fi
        bash src/tools/copy_files_to_cernbox.sh -s {config[output_path]} -d {params.cern_path}{params.output_dir} -t || echo "Warning: copy to EOS failed. Skipping remote upload."
        if [ -n "{params.email}" ]; then
            echo "Workflow for {config[label]} completed successfully on $(date)." | mail -s "Snakemake Success: {config[label]}" "{params.email}" || echo "Warning: failed to send success notification email."
        fi
        """

rule all_full_syst:
    input:
        rules.all_full_syst_F_1.input,
        rules.all_full_syst_F_2.input

rule all_full_syst_F_1_alias:
    input:
        rules.all_full_syst_F_1.input

rule all_full_syst_F_2_alias:
    input:
        rules.all_full_syst_F_2.input

localrules: final_output, all_full_syst, all_full_syst_F_1_alias, all_full_syst_F_2_alias
