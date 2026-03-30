import yaml
from src.skimmer.picoaod import PicoAOD #, fetch_metadata, resize
from coffea4bees.analysis.helpers.event_selection import apply_4b_selection
from coffea4bees.analysis.helpers.object_selection import load_object_selection_config
from src.physics.event_selection import apply_event_selection
from coffea.nanoevents import NanoEventsFactory

from src.friendtrees.FriendTreeSchema import FriendTreeSchema
from src.math_tools.random import Squares
from coffea4bees.analysis.helpers.event_weights import add_btagweights
from coffea4bees.analysis.helpers.processor_config import processor_config
from coffea4bees.analysis.helpers.event_weights import add_weights

from coffea.analysis_tools import Weights, PackedSelection
import numpy as np
from src.physics.objects.jet_corrections import apply_jerc_corrections_jsonpog
from src.physics.common import update_events
from copy import copy
import logging
import awkward as ak
import uproot

from coffea4bees.analysis.helpers.cutflow import cutflow_4b
from src.data_formats.root import Chunk
from coffea4bees.analysis.helpers.load_friend import (
    FriendTemplate,
    parse_friends,
)


class SubSampler(PicoAOD):
    def __init__(self, sub_sampling_rand_seed=5, corrections_metadata: dict = None, apply_trigWeight: bool = True, friends: dict[str, str|FriendTemplate] = None, object_selection_cfg: str = "coffea4bees/analysis/metadata/object_selection_thresholds.yml", *args, **kwargs):
        kwargs["pico_base_name"] = f'picoAOD_PSData'
        super().__init__(*args, **kwargs)

        logging.info(f"\nRunning SubSampler with these parameters: sub_sampling_rand_seed = {sub_sampling_rand_seed} args = {args}, kwargs = {kwargs}")
        # Always use cutflow_4b unless explicitly overridden
        self._cutFlow = cutflow_4b()

        self.sub_sampling_rand_seed = sub_sampling_rand_seed
        self.corrections_metadata = corrections_metadata
        self.apply_trigWeight = apply_trigWeight
        self.friends = parse_friends(friends)
        self.sel_cfg = load_object_selection_config(object_selection_cfg) if object_selection_cfg else None

    def select(self, event):

        year    = event.metadata['year']
        dataset = event.metadata['dataset']
        fname   = event.metadata['filename']
        estart  = event.metadata['entrystart']
        estop   = event.metadata['entrystop']
        processName = event.metadata['processName']
        nEvent = len(event)
        year_label = self.corrections_metadata[year]['year_label']
        chunk   = f'{dataset}::{estart:6d}:{estop:6d} >>> '
        logging.debug( f"{chunk} file is {fname}\n" )

        ### target is for new friend trees
        target = Chunk.from_coffea_events(event)


        #
        # Set process and datset dependent flags
        #
        config = processor_config(processName, dataset, event)
        logging.debug(f'{chunk} config={config}, for file {fname}\n')

        path = fname.replace(fname.split("/")[-1], "")

        #
        # Event selection
        #
        event = apply_event_selection( event, self.corrections_metadata[year], cut_on_lumimask=config["cut_on_lumimask"] )

        ## adds all the event mc weights and 1 for data
        weights, list_weight_names = add_weights( event, dataset=dataset, year_label=year_label,
                                                  corrections_metadata = self.corrections_metadata[year],
                                                  target = target,
                                                  friend_trigWeight=self.friends.get("trigWeight"),
                                                  apply_trigWeight = self.apply_trigWeight,
                                                  config=config
                                                 )

        #
        # Calculate and apply Jet Energy Calibration
        #
        if config["do_jet_calibration"]:
            jets = apply_jerc_corrections_jsonpog(event,
                                    corrections_metadata=self.corrections_metadata[self.year],
                                    isMC=config["isMC"],
                                    dataset=dataset
                                    )
        else:
            jets = event.Jet


        event = update_events(event, {"Jet": jets})

        # Apply object selection (function does not remove events, adds content to objects)
        event = apply_4b_selection( event,
                                    self.corrections_metadata[year],
                                    config=config,
                                    dataset=dataset,
                                    apply_mixeddata_sel=False,
                                    sel_cfg=self.sel_cfg,
                                   )

        selections = PackedSelection()
        selections.add( "lumimask", event.lumimask)
        selections.add( "passNoiseFilter", event.passNoiseFilter)
        selections.add( "passHLT", ( event.passHLT if config["cut_on_HLT_decision"] else np.full(len(event), True)  ) )
        selections.add( 'passJetMult',   event.passJetMult )
        selections.add( "passFourTag", event.fourTag)
        event["weight"] = weights.weight()

        cumulative_cuts = ["lumimask"]
        self._cutFlow.fill( "all",             event[selections.all(*cumulative_cuts)], allTag=True )

        other_cuts = ["passNoiseFilter", "passHLT", "passJetMult","passFourTag"]

        for cut in other_cuts:
            cumulative_cuts.append(cut)
            self._cutFlow.fill( cut, event[selections.all(*cumulative_cuts)], allTag=True )


        #
        # Calculate and apply btag scale factors
        #### AGE to add btag JES
        #
        if config["isMC"]:

            weights, list_weight_names = add_btagweights( event, weights,
                                                          list_weight_names=list_weight_names,
                                                          corrections_metadata=self.corrections_metadata[year],
                                                          isRun3=config["isRun3"],
            )
            logging.debug( f"Btag weight {weights.partial_weight(include=['CMS_btag'])[:10]}\n" )
            event["weight"] = weights.weight()

            self._cutFlow.fill( "passJetMult_btagSF", event[selections.all(*cumulative_cuts)], allTag=True )


        selev = event[selections.all(*cumulative_cuts)]

        #
        #  Pass Pseudo data subsampling...
        #
        rng = Squares("sub_sample_MC", dataset, year, self.sub_sampling_rand_seed)
        counter = np.empty((len(selev), 2), dtype=np.uint64)
        counter[:, 0] = np.asarray(selev.event).view(np.uint64)
        counter[:, 1] = np.asarray(selev.run).view(np.uint32)
        counter[:, 1] <<= 32
        counter[:, 1] |= np.asarray(selev.luminosityBlock).view(np.uint32)
        sample_rand = rng.uniform(counter, low=0, high=1.0).astype(np.float32)
        pass_sub_sample_filter_selev = (sample_rand < selev.weight)

        pass_sub_sample_filter = np.full( len(event), True)
        pass_sub_sample_filter[ selections.all(*cumulative_cuts) ] = pass_sub_sample_filter_selev
        selections.add( "passSubSample", pass_sub_sample_filter)
        cumulative_cuts.append("passSubSample")
        self._cutFlow.fill( "passSubSample", event[selections.all(*cumulative_cuts)], allTag=True )

        #logging.info(f"Weigthts are {selev.weight}\n")
        selev = event[selections.all(*cumulative_cuts)]

        out_branches = {
            # Update jets with new kinematics
            "Jet_pt":              selev.Jet.pt,
            "Jet_eta":             selev.Jet.eta,
            "Jet_phi":             selev.Jet.phi,
            }

        branches = ak.Array(out_branches)

        processOutput = {}
        processOutput["total_event"] = len(event)
        processOutput["pass_skim"]   = len(selev)

        return (selections.all(*cumulative_cuts),
                branches,
                processOutput)
