# ==============================================================================
# coffea4bees/workflows/Snakefile_bkg_syst_A_1_make_mixeddata.smk
#
# Stage A_1: Mixed-Data Production & Validation via Snakefile_MakeMixedData.smk
# ==============================================================================

if not workflow.configfiles:
    configfile: "coffea4bees/workflows/config/analysis_ttHbb_bkg_syst.yml"

config.setdefault('fourTag_use_tight', False)
config.setdefault('analysis_config', {}).setdefault('config', {}).setdefault('fourTag_use_tight', False)

pub = str(config.get('publish_base', "root://cmseos.fnal.gov//store/user/algomez/XX4b/mixeddata/Run2_v2/ttHbb_pz_rank0_0")).rstrip("/")
inputs = config.setdefault('inputs', {})
inputs.setdefault('FvT', f"{pub}/friend/FvT_nominal/result.json@@analysis.0.merged")
inputs.setdefault('JCM', f"{pub}/output/computeJCM/jetCombinatoricModel_SB.yml")
inputs.setdefault('jcm_hists', f"{pub}/output/computeJCM/histAll_NoJCM.coffea")

include: "Snakefile_MakeMixedData.smk"

rule all_bkg_syst_A_1:
    default_target: True
    input:
        rules.all_MakeMixedData.input

localrules: all_bkg_syst_A_1
