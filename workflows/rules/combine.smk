import os

include: "../helpers/common.smk"

rule workspace:
    input: "{path}.txt"
    output: "{path}.root"
    params:
        signallabel = "",
        othersignal_maps = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/workspace_{path}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        echo "$LOG"
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting workspace rule with signal {params.signallabel}" > $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            text2workspace.py $(basename {input}) \
            -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose \
            --PO 'map=.*/{params.signallabel}:r{params.signallabel}[1,-10,10]' \
            {params.othersignal_maps} \
            -o $(basename {output})" 2>&1 | tee -a $(basename $LOG)

        echo "[$(date)] Completed workspace rule with signal {params.signallabel}" >> $LOG
        """

rule limits:
    input: "{path}__{signallabel}.root"
    output: 
        txt="{path}_limits__{signallabel}.txt",
        json="{path}_limits__{signallabel}.json"
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = "",
        freeze_parameters = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/limits_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting limits rule with signal {params.signallabel}" > $LOG

        echo "[$(date)] Running AsymptoticLimits" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combine -M AsymptoticLimits $(basename {input}) \
            --redefineSignalPOIs r{params.signallabel} \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            -n _{params.signallabel}" \
            2>&1 | tee -a $LOG > {output.txt}

        echo "[$(date)] Running CollectLimits" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combineTool.py -M CollectLimits \
            higgsCombine_{params.signallabel}.AsymptoticLimits.mH120.root \
            -o $(basename {output.json})" \
            2>&1 | tee -a $LOG

        echo "[$(date)] Completed limits rule with signal {params.signallabel}" >> $LOG
        """

rule significance:
    input: "{path}__{signallabel}.root"
    output: "significance__{path}__{signallabel}.log"
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = "",
        freeze_parameters = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/significance_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting significance rule with signal {params.signallabel}" > $LOG

        echo "[$(date)] Running observed significance" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combine -M Significance $(basename {input}) \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            --redefineSignalPOIs r{params.signallabel} \
            -n _{params.signallabel}" \
            2>&1 | tee -a $LOG > {output}

        echo "[$(date)] Running expected significance" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combine -M Significance $(basename {input}) \
            --redefineSignalPOIs r{params.signallabel} \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            -n _{params.signallabel} \
            -t -1 --expectSignal=1" \
            2>&1 | tee -a $LOG >> {output}

        echo "[$(date)] Completed significance rule with signal {params.signallabel}" >> $LOG
        """

rule likelihood_scan:
    input: "{path}__{signallabel}.root"
    output: "{path}_likelihood_scan__{signallabel}.pdf"
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = "",
        freeze_parameters = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/likelihood_scan_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting likelihood_scan rule with signal {params.signallabel}" > $LOG

        echo "|---- Running initial fit"
        echo "[$(date)] Running initial fit" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combine -M MultiDimFit -d $(basename {input}) \
            -n _$(basename {input} .root)_{params.signallabel} \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            --saveWorkspace --robustFit 1" \
            2>&1 | tee -a $LOG

        echo "|---- Running MultiDimFit"
        echo "[$(date)] Running MultiDimFit" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combine -M MultiDimFit \
            -d higgsCombine_$(basename {input} .root)_{params.signallabel}.MultiDimFit.mH120.root \
            -n _$(basename {input} .root)_{params.signallabel}_final \
            -P r{params.signallabel} \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            --snapshotName MultiDimFit --rMin -10 --rMax 10 --algo grid --points 50 --alignEdges 1" \
            2>&1 | tee -a $LOG

        echo "|---- Plotting likelihood scan"
        echo "[$(date)] Plotting likelihood scan" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            plot1DScan.py higgsCombine_$(basename {input} .root)_{params.signallabel}_final.MultiDimFit.mH120.root \
            --POI r{params.signallabel} -o $(basename {output} .pdf)" \
            2>&1 | tee -a $LOG

        echo "[$(date)] Completed likelihood_scan rule with signal {params.signallabel}" >> $LOG
        """

rule impacts:
    input: "{path}__{signallabel}.root"
    output: "{path}_impacts__{signallabel}.pdf"
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = "",
        set_parameters_ranges = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/impacts_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting impacts rule with signal {params.signallabel}" > $LOG

        echo "|---- Running initial fit"
        echo "[$(date)] Running initial fit" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combineTool.py -M Impacts -d $(basename {input}) \
            --doInitialFit --robustFit 1 -m 125 \
            --setParameterRanges r{params.signallabel}=-10,10{params.set_parameters_ranges} \
            {params.set_parameters_zero} \
            -n $(basename {input} .root)" \
            2>&1 | tee -a $LOG

        echo "|---- Running fits per systematic"
        echo "[$(date)] Running fits per systematic" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combineTool.py -M Impacts -d $(basename {input}) \
            --doFits --robustFit 1 -m 125 --parallel 4 \
            --setParameterRanges r{params.signallabel}=-10,10{params.set_parameters_ranges} \
            {params.set_parameters_zero} \
            -n $(basename {input} .root)" \
            2>&1 | tee -a $LOG

        echo "|---- Running merging results"
        echo "[$(date)] Running merging results" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            combineTool.py -M Impacts \
            -m 125 -n $(basename {input} .root) \
            -d $(basename {input}) \
            -o impacts_combine_$(basename {input} .root)_exp.json" \
            2>&1 | tee -a $LOG

        echo "|---- Running creating pdf"
        echo "[$(date)] Running creating pdf" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) && \
            plotImpacts.py -i impacts_combine_$(basename {input} .root)_exp.json \
            -o $(basename {output} .pdf) \
            --POI r{params.signallabel} \
            --per-page 20 --left-margin 0.3 --height 400 --label-size 0.04" \
            2>&1 | tee -a $LOG

        echo "[$(date)] Completed impacts rule with signal {params.signallabel}" >> $LOG
        """

rule gof:
    input: "{path}__{signallabel}.root"
    output: "{path}_gof__{signallabel}.pdf"
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = "",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/gof_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting gof rule with signal {params.signallabel}" > $LOG

        echo "|---- Running Goodness of Fit tests data"
        echo "[$(date)] Running Goodness of Fit tests data" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            combine -M GoodnessOfFit $(basename {input}) \
            --algo saturated \
            {params.set_parameters_zero} \
            -n _$(basename {input} .root)_{params.signallabel}_gof_data" \
            2>&1 | tee -a $LOG $(dirname {output})/gof_data_$(basename {input} .root)_{params.signallabel}.txt

        echo "|---- Running Goodness of Fit tests toys"
        echo "[$(date)] Running Goodness of Fit tests toys" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            combine -M GoodnessOfFit $(basename {input}) \
            --toysFrequentist -t 500 --algo saturated  \
            -n _$(basename {input} .root)_{params.signallabel}_gof_toys" \
            2>&1 | tee -a $LOG $(dirname {output})/gof_toys_$(basename {input} .root)_{params.signallabel}.txt

        echo "|---- Collecting Goodness of Fit results"
        echo "[$(date)] Collecting Goodness of Fit results" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            combineTool.py -M CollectGoodnessOfFit \
            --input higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_data.GoodnessOfFit.mH120.root \
            higgsCombine_$(basename {input} .root)_{params.signallabel}_gof_toys.GoodnessOfFit.mH120.123456.root \
            -o gof_$(basename {input} .root)_{params.signallabel}.json" 2>&1 | tee -a $LOG
            
        echo "|---- Plotting Goodness of Fit results"
        echo "[$(date)] Plotting Goodness of Fit results" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            plotGof.py gof_$(basename {input} .root)_{params.signallabel}.json \
            --statistic staturated --mass 120.0 \
            --output $(basename {output} .pdf)" 2>&1 | tee -a $LOG

        echo "[$(date)] Completed gof rule with signal {params.signallabel}" >> $LOG
        """

rule postfit:
    input: "{path}__{signallabel}.root"
    output: "{path}_postfit__{signallabel}.pdf"
    params:
        signallabel = "{signallabel}",
        set_parameters_zero = "",
        freeze_parameters = "",
        channel="",
        signal="",
        ylog="",
        container_wrapper = config.get("container_wrapper", "./run_container combine")
    log: "output/logs/postfit_{path}__{signallabel}.log"
    shell:
        """
        LOG=$(pwd)/{log}
        mkdir -p $(dirname $LOG)
        echo "[$(date)] Starting postfit rule with signal {params.signallabel}" > $LOG

        echo "[$(date)] Running postfit b-only" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            combine -M FitDiagnostics $(basename {input}) \
            --redefineSignalPOIs r{params.signallabel} \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            -n _$(basename {input} .root)_prefit_bonly \
            --saveShapes --saveWithUncertainties --plots" 2>&1 | tee -a $LOG

        echo "[$(date)] Running diffNuisances for b-only" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            python /home/cmsusr/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
            -p r{params.signallabel} \
            -a fitDiagnostics_$(basename {input} .root)_prefit_bonly.root \
            -g diffNuisances_$(basename {input} .root)_prefit_bonly.root" 2>&1 | tee -a $LOG

        echo "[$(date)] Running postfit s+b" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            combine -M FitDiagnostics $(basename {input}) \
            --redefineSignalPOIs r{params.signallabel} \
            {params.set_parameters_zero} \
            {params.freeze_parameters} \
            -n _$(basename {input} .root)_prefit_sb \
            --saveShapes --saveWithUncertainties --plots" 2>&1 | tee -a $LOG

        mkdir -p $(dirname {input})/fitDiagnostics_sb/
        mv $(dirname {input})/*th1x* $(dirname {input})/fitDiagnostics_sb/ 2>/dev/null || true
        mv $(dirname {input})/covariance* $(dirname {input})/fitDiagnostics_sb/ 2>/dev/null || true

        echo "[$(date)] Running diffNuisances for s+b" >> $LOG
        {params.container_wrapper} "cd $(dirname {input}) &&\
            python /home/cmsusr/CMSSW_11_3_4/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py \
            -p r{params.signallabel} \
            -a fitDiagnostics_$(basename {input} .root)_prefit_sb.root \
            -g diffNuisances_$(basename {input} .root)_prefit_sb.root" 2>&1 | tee -a $LOG

        echo "[$(date)] Running postfit plots for b-only" >> $LOG

        {params.container_wrapper} \
            python3 coffea4bees/plots/make_postfit_plot.py \
                -i $(dirname {input})/fitDiagnostics_$(basename {input} .root)_prefit_sb.root \
                -o $(dirname {input})/plots/ \
                -c {params.channel} \
                -s {params.signal} \
                --log {params.ylog} \
                -m coffea4bees/stats_analysis/metadata/{params.channel}.yml

        echo "[$(date)] Completed postfit rule with signal {params.signallabel}" >> $LOG
        """