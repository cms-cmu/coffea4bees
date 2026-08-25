"""
Categorization of ttbar events by additional heavy-flavour jets, using the
``genTtbarId`` branch.

Reference:
https://twiki.cern.ch/twiki/bin/view/CMSPublic/GenHFHadronMatcher#Event_categorization_example_1

The three categories are mutually exclusive and exhaustive:

===========  ==========================  ===================================
category     ``abs(genTtbarId) % 100``   meaning
===========  ==========================  ===================================
``ttB``      51, 52, 53, 54, 55          extra b jets  (tt+b, tt+2b, tt+bb)
``ttC``      41, 42, 43, 44, 45          extra c jets  (tt+c, tt+2c, tt+cc)
``ttLF``     anything else               no extra heavy flavour
===========  ==========================  ===================================

``abs()`` is applied before the modulo on purpose: for a negative ``genTtbarId``
Python/NumPy ``%`` returns a non-negative result that does *not* correspond to
the category encoding (e.g. ``-51 % 100 == 49``), so the two spellings are not
interchangeable.

The helpers work on anything supporting ``abs``, ``%`` and ``==`` elementwise,
i.e. both NumPy and Awkward arrays.
"""

TTB_IDS = (51, 52, 53, 54, 55)
TTC_IDS = (41, 42, 43, 44, 45)

CATEGORIES = ("ttB", "ttC", "ttLF")


def _mod100(genTtbarId):
    return abs(genTtbarId) % 100


def _any_of(mod, ids):
    mask = mod == ids[0]
    for i in ids[1:]:
        mask = mask | (mod == i)
    return mask


def is_ttB(genTtbarId):
    """Events with additional b jets (``abs(genTtbarId) % 100`` in 51..55)."""
    return _any_of(_mod100(genTtbarId), TTB_IDS)


def is_ttC(genTtbarId):
    """Events with additional c jets (``abs(genTtbarId) % 100`` in 41..45)."""
    return _any_of(_mod100(genTtbarId), TTC_IDS)


def is_ttLF(genTtbarId):
    """Events with no additional heavy-flavour jets (neither ttB nor ttC)."""
    mod = _mod100(genTtbarId)
    return ~(_any_of(mod, TTB_IDS) | _any_of(mod, TTC_IDS))


def ttbar_category_mask(genTtbarId, category: str):
    """Return the boolean mask for ``category`` (one of :data:`CATEGORIES`).

    ``"notB"`` is also accepted as a convenience for ``ttC | ttLF``, which is
    the complement of ``ttB`` and is what the tt+bb stitching keeps from the
    inclusive ttbar sample.
    """
    if category == "ttB":
        return is_ttB(genTtbarId)
    if category == "ttC":
        return is_ttC(genTtbarId)
    if category == "ttLF":
        return is_ttLF(genTtbarId)
    if category == "notB":
        return ~is_ttB(genTtbarId)
    raise ValueError(
        f"unknown ttbar category {category!r}; expected one of {CATEGORIES + ('notB',)}"
    )
