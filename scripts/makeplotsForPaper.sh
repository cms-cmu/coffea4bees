python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/mixedVsDataVs3b/RunII/passPreSel/fourTag/SR/SvB_MA_noFvT_ps_hh_ave_logy_fixed_binning.yaml   -o plotsForPaper_output/mixedVsDataVs3b
/bin/mv plotsForPaper_output/mixedVsDataVs3b/RunII/passPreSel/region_SR/SvB_MA_noFvT_ps_hh_ave_fixed_binning_logy.pdf  plotsForPaper_output/mixedVsDataVs3b_region_SR_SvB_MA_noFvT_ps_hh_ave_fixed_binning_logy.pdf 
python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/mixedVsDataVs3b/RunII/passPreSel/fourTag/SB/SvB_MA_noFvT_ps_hh_ave_logy_fixed_binning.yaml   -o plotsForPaper_output/mixedVsDataVs3b
/bin/mv plotsForPaper_output/mixedVsDataVs3b/RunII/passPreSel/region_SB/SvB_MA_noFvT_ps_hh_ave_fixed_binning_logy.pdf  plotsForPaper_output/mixedVsDataVs3b_region_SB_SvB_MA_noFvT_ps_hh_ave_fixed_binning_logy.pdf 

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/RunII/passPreSel/fourTag/SB/SvB_MA_ps_hh_logy_fixed_binning.yaml   -o plotsForPaper_output
/bin/mv plotsForPaper_output/RunII/passPreSel/region_SB/SvB_MA_ps_hh_logy.pdf plotsForPaper_output/region_SB_SvB_MA_ps_hh_logy.pdf 
python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/RunII/passPreSel/fourTag/SB/SvB_MA_noFvT_ps_hh_logy_fixed_binning.yaml  -o plotsForPaper_output
/bin/mv plotsForPaper_output/RunII/passPreSel/region_SB//SvB_MA_noFvT_ps_hh_logy.pdf plotsForPaper_output/region_SB_SvB_MA_noFvT_ps_hh_logy.pdf

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_prefit_138.yaml   -o plotsForPaper_output/
/bin/mv plotsForPaper_output//RunII/passPreSel//SvB_MA_postfitplots_prefit_138_logy.pdf  plotsForPaper_output/SvB_MA_postfitplots_prefit_138_logy.pdf

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b_138.yaml   -o plotsForPaper_output/
/bin/mv plotsForPaper_output/RunII/passPreSel/SvB_MA_ps_hh_logy_postfit_fit_b_138_logy.pdf  plotsForPaper_output//SvB_MA_ps_hh_logy_postfit_fit_b_138_logy.pdf

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_s_138.yaml   -o plotsForPaper_output/
/bin/mv plotsForPaper_output/RunII/passPreSel/SvB_MA_ps_hh_logy_postfit_fit_s_138_logy.pdf  plotsForPaper_output//SvB_MA_ps_hh_logy_postfit_fit_s_138_logy.pdf

# python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_b.yaml   -o plotsForPaper_output/
# python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_prefit.yaml   -o plotsForPaper_output/
# python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/SvB_MA_postfitplots_fit_s.yaml   -o plotsForPaper_output/

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/Mixeddata_SvB_MA_postfitplots_fit_b.yaml   -o plotsForPaper_output/
/bin/mv plotsForPaper_output/RunII/passPreSel/Mixeddata_SvB_MA_postfitplots_fit_b_logy.pdf  plotsForPaper_output/Mixeddata_SvB_MA_postfitplots_fit_b_logy.pdf

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/Mixeddata_SvB_MA_postfitplots_prefit.yaml   -o plotsForPaper_output/
/bin/mv plotsForPaper_output/RunII/passPreSel/Mixeddata_SvB_MA_postfitplots_prefit_logy.pdf plotsForPaper_output/Mixeddata_SvB_MA_postfitplots_prefit_logy.pdf

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/Mixeddata_SvB_MA_postfitplots_fit_s.yaml   -o plotsForPaper_output/
/bin/mv plotsForPaper_output/RunII/passPreSel/Mixeddata_SvB_MA_ps_hh_logy_postfit_fit_s_logy.pdf plotsForPaper_output//Mixeddata_SvB_MA_ps_hh_logy_postfit_fit_s_logy.pdf


python coffea4bees/plots/variance_plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/0_variance_multijet_ensemble_basis0.yaml   -o plotsForPaper_output/
python coffea4bees/plots/variance_plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/0_variance_multijet_ensemble_basis1.yaml   -o plotsForPaper_output/
python coffea4bees/plots/variance_plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/0_variance_multijet_ensemble_basis2.yaml   -o plotsForPaper_output/
python coffea4bees/plots/variance_plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/0_variance_multijet_ensemble_basis3.yaml   -o plotsForPaper_output/
python coffea4bees/plots/variance_plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/0_variance_multijet_ensemble_basis4.yaml   -o plotsForPaper_output/
python coffea4bees/plots/variance_plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/0_variance_multijet_ensemble_basis5.yaml   -o plotsForPaper_output/

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/1_bias_basis-1.yaml    -o plotsForPaper_output/ 
/bin/mv plotsForPaper_output/RunII/passPreSel/1_bias_basis-1_logy.pdf plotsForPaper_output/1_bias_basis-1_logy.pdf

python coffea4bees/plots/plot_from_yaml.py --input_yaml_files coffea4bees/archive/plotsForPaper/1_bias_basis0.yaml    -o plotsForPaper_output/
/bin/mv plotsForPaper_output/RunII/passPreSel/1_bias_basis0_logy.pdf plotsForPaper_output/1_bias_basis0_logy.pdf
