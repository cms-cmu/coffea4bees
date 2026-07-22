from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.types as tt
import fsspec

from src.classifier.nn.blocks.HCR import HCR as BaseHCR, InputEmbed as BaseInputEmbed
from src.classifier.ml.models.HCR import HCRModel as BaseHCRModel, HCRTraining as BaseHCRTraining, HCRModelEval as BaseHCRModelEval, HCREvaluation as BaseHCREvaluation
from src.classifier.config.setting.HCR import Input, InputBranch, Output
from src.classifier.config.state.label import MultiClass
from src.classifier.ml.models.HCR import GBNSchedule, HCRBenchmarks, _HCRInput, _HCRSkim, HCRArch
from src.classifier.ml.evaluation import EvaluationStage
from src.classifier.config.setting.ml import SplitterKeys
from src.classifier.nn.dataset import simple_loader, subset
from src.classifier.config.scheduler import SkimStep
from src.classifier.ml.training import TrainingStage, BenchmarkStage, OutputStage
from src.classifier.utils import MemoryViewIO

if TYPE_CHECKING:
    from src.classifier.ml import BatchType
    from src.classifier.ml.skimmer import Splitter
    from src.storage.eos import PathLike


def get_ancillary_tensors(dataset):
    """Retrieve the raw ancillary tensor from a NamedTensorDataset or Subset wrapper."""
    if hasattr(dataset, "datasets") and isinstance(dataset.datasets, dict) and "ancillary" in dataset.datasets:
        return dataset.datasets["ancillary"]
    elif hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        underlying = get_ancillary_tensors(dataset.dataset)
        if underlying is not None:
            return underlying[dataset.indices]
    return None


class InputEmbed_lowpt(BaseInputEmbed):
    def __init__(self, *args, offsets, **kwargs):
        super().__init__(*args, **kwargs)
        self.offsets = offsets
        self._nSelJets_idx = None  # prevent parent from running hardcoded log transform

    def dataPrep(self, j, o, a):
        device = j.get_device() if j.get_device() >= 0 else "cpu"
        n = j.shape[0]
        j = j.view(n, 4, 4)
        o = o.view(n, 5, -1)
        a = a.view(n, self.dA, 1)

        # Apply log transform dynamically using our offsets buffer:
        for idx, name in enumerate(self.ancillaryFeatures):
            if "SelJets" in name or "Jet" in name:
                a[:, idx, :] = torch.log(a[:, idx, :] - self.offsets[idx])

        if self.store:
            self.storeData["canJets"] = j.detach().to("cpu").numpy()
            self.storeData["otherJets"] = o.detach().to("cpu").numpy()

        # make leading jet eta positive direction so detector absolute eta info is removed
        etaSign = (
            1 - 2 * (j[:, 1, 0:1] < 0).float()
        )  # -1 if eta is negative, +1 if eta is zero or positive
        j[:, 1, :] = etaSign * j[:, 1, :]

        from src.classifier.nn.blocks.HCR import addFourVectors, PxPyPzE, matrixMdPhi, calcDeltaPhi, NonLU, isinf

        d, dPxPyPzE = addFourVectors(
            j[:, :, (0, 2, 0, 1, 0, 1)], j[:, :, (1, 3, 2, 3, 3, 2)]
        )

        q, qPxPyPzE = addFourVectors(
            d[:, :, (0, 2, 4)],
            d[:, :, (1, 3, 5)],
            v1PxPyPzE=dPxPyPzE[:, :, (0, 2, 4)],
            v2PxPyPzE=dPxPyPzE[:, :, (1, 3, 5)],
        )

        # do data prep for the other jets if we are using them
        mask, ooMdPhi, doMdPhi, mask_oo, mask_do = None, None, None, None, None
        if self.useOthJets:
            o[:, 1, :] = etaSign * o[:, 1, :]
            j_isCanJet = torch.cat(
                [j, 2 * torch.ones((n, 1, 4), dtype=torch.float).to(device)], 1
            )  # label canJets with 2 (-1 for mask, 0 for not preselected, 1 for preselected jet)
            o = torch.cat([j_isCanJet, o], 2)
            mask = (o[:, 4, :] == -1).to(device)
            oPxPyPzE = PxPyPzE(o)

            n = d.shape[0]
            # compute matrix of dijet masses and opening angles between other jets
            ooMdPhi = matrixMdPhi(o, o, v1PxPyPzE=oPxPyPzE, v2PxPyPzE=oPxPyPzE)
            ooMdPhi = torch.cat(
                [
                    ooMdPhi,
                    torch.zeros((n, 1, self.osl, self.osl), dtype=torch.float).to(
                        device
                    ),
                ],
                1,
            )  # flag with zeros to signify dijet quantities

            mask_oo = mask.view(n, 1, self.osl) | mask.view(
                n, self.osl, 1
            )  # mask of 2d matrix of otherjets (i,j) is True if mask[i] | mask[j]
            mask_oo = mask_oo.masked_fill(self.mask_oo_same.to(device), 1)

            # compute matrix of trijet masses and opening angles between dijets and other jets
            doMdPhi = matrixMdPhi(d, o, v1PxPyPzE=dPxPyPzE, v2PxPyPzE=oPxPyPzE)
            doMdPhi = torch.cat(
                [
                    doMdPhi,
                    torch.ones((n, 1, self.dsl, self.osl), dtype=torch.float).to(
                        device
                    ),
                ],
                1,
            )  # flag with ones to signify trijet quantities

            mask_do = mask.view(n, 1, self.osl).repeat(1, self.dsl, 1)
            # repeat so we can change mask for each dijet
            mask_do = mask_do.masked_fill(self.mask_do_same.to(device), 1)

            o[:, (0, 3), :] = torch.log(1 + o[:, (0, 3), :])
            o[isinf(o)] = -1  # isinf not supported by ONNX

            o = torch.cat(
                (o[:, :2, :], o[:, 3:, :]), 1
            )  # remove phi from othJet features

        j[:, (0, 3), :] = torch.log(1 + j[:, (0, 3), :])
        d[:, (0, 3), :] = torch.log(1 + d[:, (0, 3), :])
        q[:, (0, 3), :] = torch.log(1 + q[:, (0, 3), :])

        j = torch.cat([j, j[:, :, (0, 2, 1, 3)], j[:, :, (0, 3, 1, 2)]], 2)
        # only keep relative angular information so that learned features are invariant under global phi rotations and eta/phi flips
        j[:, 2:3, (0, 2, 4, 6, 8, 10)] = calcDeltaPhi(
            d, j[:, :, (0, 2, 4, 6, 8, 10)]
        )  # replace jet phi with deltaPhi between dijet and jet
        j[:, 2:3, (1, 3, 5, 7, 9, 11)] = calcDeltaPhi(d, j[:, :, (1, 3, 5, 7, 9, 11)])

        d[:, 2:3, (0, 2, 4)] = calcDeltaPhi(q, d[:, :, (0, 2, 4)])
        d[:, 2:3, (1, 3, 4)] = calcDeltaPhi(q, d[:, :, (1, 3, 5)])

        q = torch.cat((q[:, :2, :], q[:, 3:, :]), 1)  # remove phi from quadjet features

        return j, d, q, a, o, ooMdPhi, doMdPhi, mask, mask_oo, mask_do


class HCR_lowpt(BaseHCR):
    def __init__(self, dijetFeatures, quadjetFeatures, ancillaryFeatures, useOthJets="", device="cuda", nClasses=1, architecture="HCR"):
        super(BaseHCR, self).__init__()
        self.debug = False
        self.dA = len(ancillaryFeatures)
        self.dD = dijetFeatures
        self.dQ = quadjetFeatures
        self.device = device
        self.name = architecture + ("+" + useOthJets if useOthJets else "") + "_%d" % (dijetFeatures)
        self.useOthJets = bool(useOthJets)
        self.nC = nClasses
        self.store = None
        self.storeData = {}
        self.onnx = False
        self.nGhostBatches = 64
        self.phase_symmetric = True

        from src.classifier.nn.blocks.HCR import layerOrganizer, ResNetBlock, MinimalAttention, GhostBatchNorm1d

        self.layers = layerOrganizer()

        # Register dynamic offsets buffers
        self.register_buffer("offsets", torch.zeros(len(ancillaryFeatures)))
        self.register_buffer("offsets_initialized", torch.tensor(False))

        # Use InputEmbed_lowpt
        self.inputEmbed = InputEmbed_lowpt(
            dijetFeatures=self.dD,
            quadjetFeatures=self.dQ,
            ancillaryFeatures=ancillaryFeatures,
            useOthJets=self.useOthJets,
            layers=self.layers,
            device=self.device,
            phase_symmetric=self.phase_symmetric,
            offsets=self.offsets,
        )

        self.dijetResNetBlock = ResNetBlock(
            self.dD,
            prefix="",
            nLayers=2,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.jetConv, self.inputEmbed.dijetConv],
        )
        previousLayer = self.dijetResNetBlock.reinforce[-1]
        if self.useOthJets:
            self.attention_oo = MinimalAttention(
                self.dD,
                heads=2,
                phase_symmetric=self.phase_symmetric,
                layers=self.layers,
                inputLayers=[self.inputEmbed.othJetConv],
                device=self.device,
            )
            self.attention_do = MinimalAttention(
                self.dD,
                heads=2,
                phase_symmetric=self.phase_symmetric,
                layers=self.layers,
                inputLayers=[self.dijetResNetBlock.reinforce[-1]],
                device=self.device,
            )
            previousLayer = self.attention_do.conv

        self.layers.addLayer(
            self.inputEmbed.quadjetEmbed, startIndex=previousLayer.index - 1
        )
        self.layers.addLayer(
            self.inputEmbed.quadjetConv, [self.inputEmbed.quadjetEmbed]
        )

        self.quadjetResNetBlock = ResNetBlock(
            self.dQ,
            prefix="di",
            nLayers=2,
            xx0Update=False,
            phase_symmetric=self.phase_symmetric,
            device=self.device,
            layers=self.layers,
            inputLayers=[self.inputEmbed.quadjetConv, previousLayer],
        )

        self.select_q = GhostBatchNorm1d(
            self.dQ,
            features_out=1,
            conv=True,
            bias=False,
            name="quadjet selector",
            device=self.device,
        )
        self.out = GhostBatchNorm1d(
            self.dQ, features_out=self.nC, conv=True, name="out", device=self.device
        )

        self.layers.addLayer(self.select_q, [self.quadjetResNetBlock.reinforce[-1]])
        self.layers.addLayer(
            self.out, [self.select_q]
        )
        self.forwardCalls = 0


class HCRModel_lowpt(BaseHCRModel):
    def __init__(self, device: tt.Device, arch: HCRArch, benchmarks: HCRBenchmarks):
        self._loss = arch.loss
        self._device = device
        self._gbn = None
        self._arch = arch
        self._nn = HCR_lowpt(
            dijetFeatures=arch.n_features,
            quadjetFeatures=arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            useOthJets=("attention" if arch.attention else ""),
            device=device,
            nClasses=MultiClass.n_trainable(),
        )
        self._benchmarks = benchmarks


class HCRTraining_lowpt(BaseHCRTraining):
    def setup_dynamic_offsets(self):
        ancillary_tensor = get_ancillary_tensors(self.dataset)
        if ancillary_tensor is not None:
            min_vals = ancillary_tensor.min(dim=0).values.cpu()
            offsets = torch.zeros_like(min_vals)
            feature_names = self._HCR.nn.inputEmbed.ancillaryFeatures
            for idx, name in enumerate(feature_names):
                if "SelJets" in name or "Jet" in name:
                    offsets[idx] = min_vals[idx] - 1
            # Write to module's registered buffer
            self._HCR.nn.offsets.copy_(offsets.to(self.device))
            self._HCR.nn.offsets_initialized.copy_(torch.tensor(True).to(self.device))
            logging.info(f"[HCR_lowpt] Dynamic offsets initialized: {dict(zip(feature_names, offsets.tolist()))}")
        else:
            logging.warning("[HCR_lowpt] Could not retrieve ancillary tensor from dataset to compute dynamic offsets.")

    def stages(self):
        self._HCR = HCRModel_lowpt(
            device=self.device,
            arch=self._arch,
            benchmarks=self._benchmarks,
        )
        self._HCR.ghost_batch = self._ghost_batch
        self._HCR.to(self.device)

        # Perform dynamic offset scanning before training/initialization starts
        self.setup_dynamic_offsets()

        if self._pretrained_weights is not None:
            if self._pretrained_weights.endswith(".pkl"):
                path = self._pretrained_weights
            else:
                import glob as _glob
                offset = self.metadata.get("offset", 0)
                pattern = f"{self._pretrained_weights}/*_offset{offset}_*.pkl"
                matches = _glob.glob(pattern)
                if not matches:
                    raise FileNotFoundError(
                        f"No pretrained weights found for offset {offset} in "
                        f"{self._pretrained_weights}. Pattern: {pattern}"
                    )
                if len(matches) > 1:
                    raise FileNotFoundError(
                        f"Ambiguous pretrained weights for offset {offset} in "
                        f"{self._pretrained_weights}: {matches}. "
                        f"Use a more specific directory or pass a direct .pkl path."
                    )
                path = matches[0]
            logging.info(f"Warm-starting from {path}")
            with fsspec.open(path, "rb") as f:
                saved = torch.load(f, map_location=self.device, weights_only=False)
            self._HCR.nn.load_state_dict(saved["model"])

        self._splitter.setup(self.dataset)
        skim = _HCRSkim(self._HCR._nn, self.device, self._splitter)
        yield TrainingStage(
            name="Initialization",
            model=skim,
            schedule=SkimStep(),
            training=self.dataset,
        )
        self._HCR.nn.initMeanStd()
        validation_sets = self._splitter.get()
        training_set = validation_sets[SplitterKeys.training]
        yield BenchmarkStage(
            name="Baseline",
            model=self._HCR,
            validation=validation_sets,
        )
        yield TrainingStage(
            name="Training",
            model=self._HCR,
            schedule=self._training,
            training=training_set,
            validation=validation_sets,
        )
        self._HCR.ghost_batch = None
        if self._finetuning is not None:
            layers = self._HCR._nn.layers
            layers.setLayerRequiresGrad(
                requires_grad=False, index=self._HCR._nn.embedding_layers()
            )
            yield TrainingStage(
                name="Finetuning",
                model=self._HCR,
                schedule=self._finetuning,
                training=training_set,
                validation=validation_sets,
            )
            self._HCR.ghost_batch = self._ghost_batch
            layers.setLayerRequiresGrad(requires_grad=True)
        
        output_stage = OutputStage(name="Final", path=f"{self.name}__{self.uuid}.pkl")
        output_path = output_stage.absolute_path
        if not output_path.is_null:
            logging.info(f"Saving model to {output_path}")
            with fsspec.open(output_path, "wb") as f:
                torch.save(
                    {
                        "model": self._HCR.nn.state_dict(),
                        "metadata": self.metadata,
                        "uuid": self.uuid,
                        "label": MultiClass.trainable_labels,
                        "arch": self._arch.save(),
                        "input": {
                            k: getattr(InputBranch, k)
                            for k in (
                                "feature_ancillary",
                                "feature_CanJet",
                                "feature_NotCanJet",
                                "n_NotCanJet",
                            )
                        },
                    },
                    MemoryViewIO(f),
                )
            yield output_stage


class HCRModelEval_lowpt(BaseHCRModelEval):
    def __init__(self, device: tt.Device, saved: dict[str], splitter: Splitter, mapping: Callable[[BatchType], BatchType]):
        self._device = device
        self._splitter = splitter
        self._mapping = mapping
        self._classes = saved["label"]
        for k in saved["input"].keys():
            if getattr(InputBranch, k) != saved["input"][k]:
                raise ValueError(
                    f'Input features "{k}" mismatch: training={saved["input"][k]}, evaluation={getattr(InputBranch, k)}'
                )
        self._arch = HCRArch.load(saved["arch"])
        self._nn = HCR_lowpt(
            dijetFeatures=self._arch.n_features,
            quadjetFeatures=self._arch.n_features,
            ancillaryFeatures=InputBranch.feature_ancillary,
            useOthJets=("attention" if self._arch.attention else ""),
            device=device,
            nClasses=len(self._classes),
        )
        self._nn.load_state_dict(saved["model"])

    @property
    def nn(self):
        return self._nn


class HCREvaluation_lowpt(BaseHCREvaluation):
    def stages(self):
        with fsspec.open(self._model, "rb") as f:
            load_kw = {}
            if self.device.type == "cpu":
                load_kw["map_location"] = torch.device("cpu")
            saved = torch.load(f, **load_kw, weights_only=False)
        self._HCR = HCRModelEval_lowpt(
            device=self.device,
            saved=saved,
            splitter=self._splitter,
            mapping=self._mapping,
        )
        self._HCR.to(self.device)
        yield EvaluationStage(
            name="Evaluation",
            model=self._HCR,
            dataset=self.dataset,
            dumper_kwargs={"name": self.name},
        )
