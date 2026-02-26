from coffea4bees.analysis.processors.processor_HH4b import HH4bBaseProcessor

from src.hist_tools import Collection, Fill
from coffea4bees.analysis.helpers.hist_templates import (
    QuadJetHistsSelected,
    QuadJetHistsMinDr,
    QuadJetHistsSRSingle,
    SvBHists,
)
import logging

def filling_FSR_recovery_histograms(
        selev,
        JCM,
        processName: str = None,
        year: str = 'UL18',
        isMC: bool = False,
        histCuts: list = [],
        apply_FvT: bool = False,
        run_SvB: bool = False,
        top_reconstruction: bool = False,
        isDataForMixed: bool = False,
        tag_list: list = ["threeTag", "fourTag"],
        run_dilep_ttbar_crosscheck: bool = False,
        event_metadata: dict = {},
        weight_name = "weight"
        ):

    fill = Fill(process=processName, year=year, weight=weight_name)

    hist = Collection(
        process=[processName],
        year=[year],
        tag=["fourTag"],
        region=['SR', "SB"],
        **dict((s, ...) for s in histCuts)
    )

    fill += hist.add("trigWeight", (40, 0, 2, ("trigWeight", 'Trigger weight')), weight='no_weight')

    fill += QuadJetHistsSelected(("quadJet_selected", "Selected Quad Jet"), "quadJet_selected")
    #fill += QuadJetHistsMinDr(("quadJet_min_dr", "Min dR Quad Jet"), "quadJet_min_dr")

    # fill histograms
    fill(selev, hist)
    return hist.to_dict(nonempty=True)



class analysis(HH4bBaseProcessor):
    def __init__(
        self,
        **kwargs  # Accept additional arguments to pass to parent
    ):
        # Initialize parent without JCM (we'll handle it ourselves)
        super().__init__(**kwargs)



    def custom_processing(self, selev, config):
        logging.info(f"processor_FSR_recovery.custom_processing")


        # quadJet_selected.subl is the subleading H->bb candidate
        print(f" Hbb lead Mass\n")
        print(f" lead.mass {selev.quadJet_selected.lead.mass[0:10]}\n")


        # quadJet_selected.subl is the subleading H->bb candidate
        print(f"Hbb Subl Mass\n")
        print(f" subl.mass {selev.quadJet_selected.subl.mass[0:10]}\n")


        # quadJet_selected.subl.lead is the leading jet of subleading H->bb candidate
        print(f" Hbb subl.lead\n")
        print(f"\t pt {selev.quadJet_selected.subl.lead.pt[0:10]} \n\t eta: {selev.quadJet_selected.subl.lead.eta[0:10]} \n\t phi: {selev.quadJet_selected.subl.lead.phi}\n")

        # quadJet_selected.subl.subl is the sublleading jet of subleading H->bb candidate
        print(f"Hbb subl.subl\n")
        print(f"\t pt {selev.quadJet_selected.subl.subl.pt[0:10]} \n\t eta: {selev.quadJet_selected.subl.subl.eta[0:10]} \n\t phi: {selev.quadJet_selected.subl.subl.phi}\n")


        # Event Selection where FRS likely
        # lead~125 / subl < 125


        return selev


    def histograms(self, event, selev, weights, analysis_selections, shift_name):
        """Fill histograms for analysis.

        Requires chunk-scoped variables: processName, year, config
        Must be called after process() has initialized these variables.

        Args:
            event: Event array
            selev: Selected events array
            weights: Weights object
            analysis_selections: Boolean mask for analysis selection
            shift_name: Name of systematic shift (None for nominal)

        Returns:
            Dictionary with histogram outputs
        """
        print("\n processor_FSR_recovery.histogram \n")


        # Need to figure this out...

        return filling_FSR_recovery_histograms(selev,
                                               self.apply_JCM,
                                               processName=self.processName,
                                               year=self.year,
                                               isMC=self.config["isMC"],
                                               histCuts=self.histCuts,
                                               apply_FvT=self.apply_FvT,
                                               run_SvB=self.run_SvB,
                                               run_dilep_ttbar_crosscheck=self.run_dilep_ttbar_crosscheck,
                                               top_reconstruction=self.top_reconstruction,
                                               isDataForMixed=self.config['isDataForMixed'],
                                               event_metadata=event.metadata
                                               )
