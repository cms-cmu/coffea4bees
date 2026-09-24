from enum import IntEnum

from src.classifier.task import GlobalSetting

# Exported names. REQUIRED: src/classifier/config/setting/HCR.py is a compat shim
# that does `from coffea4bees...setting.HCR import *` then `import __all__`. Without
# this list that second import raises ImportError, the shim falls back to its stale
# local definitions (feature_CanJet=["pt","eta","phi","mass"]), and any module that
# imports InputBranch from src.* (e.g. the HCR model) silently gets the 4-feature
# defaults instead of the configured features -> "shape '[N,4,4]' is invalid".
__all__ = ["InputBranch", "Input", "Output", "MassRegion", "NTag"]


class InputBranch(GlobalSetting):
    "Name of branches in the input root file"

    feature_CanJet: list[str] = ["pt", "eta", "phi", "mass"]
    feature_NotCanJet: list[str] = feature_CanJet + ["isSelJet"]
    feature_ancillary: list[str] = ["year", "nSelJets", "xW", "xbW"]
    n_CanJet: int = 4
    n_NotCanJet: int = 8
    pad_value: float = -1

    # Which MassRegion flags are folded into the region index. Empty = every member, which is
    # the Run 2 behaviour and stays the default.
    #
    # The index is an OR of MassRegion values, which is only well defined when the channel
    # flags decompose SR -- true for Run 2, where SR = ZZSR|ZHSR|HHSR and SB = ~SR. Run 3 takes
    # SR from the radial distance rhh and leaves ZZSR/ZHSR/HHSR as independent mass windows, so
    # an event can be in the ZZ window *and* the Run 3 sideband: ZZSR|SB = 7, not a member.
    #
    # Set ["SR", "SB"] for Run 3. The friend trees still carry the channel columns -- they are
    # simply not encoded, so no reprocessing is needed and they are available again as soon as
    # Run 3 ZZ/ZH samples make the decomposition meaningful. Nothing in Run 3 reads them today:
    # the FvT selects on (SB | SR) and the SvB on --regions, which defaults to ["SR"].
    mass_regions: list[str] = []

    @classmethod
    def get__mass_regions(cls, var: list[str]):
        # MassRegion is defined further down this module; this only runs at config time, long
        # after import, so the module-level name resolves.
        names = {m.name for m in MassRegion}
        unknown = [r for r in var if r not in names]
        if unknown:
            raise ValueError(
                f"InputBranch.mass_regions: unknown region(s) {unknown}; "
                f"valid names are {sorted(names)}"
            )
        return list(var)

    @classmethod
    def get__feature_CanJet(cls, var: list[str]):
        return [f"CanJet_{f}" for f in var]

    @classmethod
    def get__feature_NotCanJet(cls, var: list[str]):
        return [f"NotCanJet_{f}" for f in var]

    @classmethod
    def get__feature_ancillary(cls, var: list[str]):
        return var.copy()


class Input(GlobalSetting):
    "Name of the keys in the input batch."

    label: str = "label"
    region: str = "region"
    weight: str = "weight"
    ancillary: str = "ancillary"
    CanJet: str = "CanJet"
    NotCanJet: str = "NotCanJet"


class Output(GlobalSetting):
    "Name of the keys in the output batch."

    class_raw: str = "class_raw"
    class_prob: str = "class_prob"
    quadjet_raw: str = "quadjet_raw"
    quadjet_prob: str = "quadjet_prob"


class MassRegion(IntEnum):
    SB = 0b10
    ZZSR = 0b0101
    ZHSR = 0b1001
    HHSR = 0b1101
    SR = 0b01


class NTag(IntEnum):
    fourTag = 0b10
    threeTag = 0b01
