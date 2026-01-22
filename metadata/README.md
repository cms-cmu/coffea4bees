# Info about metadata

## In HH4b

### Run 2

- `datasets_HH4b_Run2` contains datasets for Run2 analysis, including trigger weights. The nomenclature for the folders inside:
  - Data, TTbar and mixdata samples are from the C++ code. The different in versions are only in signal samples.
  - **`2024_v1p2` is the version used in HIG-24-011.** Use this one for any Run2 analysis.
  - `2024_v1p1` is the first pico production from this code, but without removing the events with huge genweights. 
  - `2024_v1` is a mix of datasets between the old C++ framework and the first pico from this code. Mainly for archives
  - `2024_v2` is the one where all the picos are produced with the new code, but it was not used in any analysis.

- Files in '/store/user/algomez/XX4b/20231115/' and '/store/user/jda102/condor/ZH4b/ULTrig/' were produced before Nov 2023 and with the old framework.

### Run 3

- `datasets_HH4b_Run3` contains datasets for Run3 analysis. It is still under development.