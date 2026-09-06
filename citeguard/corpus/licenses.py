"""License classification and validation for corpus documents.

Unknown or proprietary licenses are treated conservatively — documents
must explicitly opt-in via an open license to be included in the corpus.
"""

from __future__ import annotations

from enum import Enum


class LicenseType(str, Enum):
    """License categories for corpus inclusion decisions."""

    CC_BY = "CC-BY"
    CC_BY_SA = "CC-BY-SA"
    CC_BY_NC = "CC-BY-NC"
    CC_BY_NC_SA = "CC-BY-NC-SA"
    CC_BY_ND = "CC-BY-ND"
    CC_BY_NC_ND = "CC-BY-NC-ND"
    CC0 = "CC0"
    PUBLIC_DOMAIN = "public_domain"
    OPEN_ACCESS = "open_access"
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"


# Licenses that allow redistribution and derivative use in a corpus.
_OPEN_LICENSES: frozenset[LicenseType] = frozenset({
    LicenseType.CC_BY,
    LicenseType.CC_BY_SA,
    LicenseType.CC_BY_NC,
    LicenseType.CC_BY_NC_SA,
    LicenseType.CC0,
    LicenseType.PUBLIC_DOMAIN,
})

# ND licenses allow sharing but not derivatives — conservative exclusion.
_RESTRICTED_LICENSES: frozenset[LicenseType] = frozenset({
    LicenseType.CC_BY_ND,
    LicenseType.CC_BY_NC_ND,
    LicenseType.PROPRIETARY,
    LicenseType.UNKNOWN,
})

# Normalized variants of common license strings.
_LICENSE_ALIASES: dict[str, LicenseType] = {
    "cc-by": LicenseType.CC_BY,
    "cc by": LicenseType.CC_BY,
    "cc-by-4.0": LicenseType.CC_BY,
    "cc-by-sa": LicenseType.CC_BY_SA,
    "cc-by-sa-4.0": LicenseType.CC_BY_SA,
    "cc-by-nc": LicenseType.CC_BY_NC,
    "cc-by-nc-4.0": LicenseType.CC_BY_NC,
    "cc-by-nc-sa": LicenseType.CC_BY_NC_SA,
    "cc-by-nc-sa-4.0": LicenseType.CC_BY_NC_SA,
    "cc-by-nd": LicenseType.CC_BY_ND,
    "cc-by-nd-4.0": LicenseType.CC_BY_ND,
    "cc-by-nc-nd": LicenseType.CC_BY_NC_ND,
    "cc-by-nc-nd-4.0": LicenseType.CC_BY_NC_ND,
    "cc0": LicenseType.CC0,
    "cc0-1.0": LicenseType.CC0,
    "public domain": LicenseType.PUBLIC_DOMAIN,
    "pd": LicenseType.PUBLIC_DOMAIN,
    "open access": LicenseType.OPEN_ACCESS,
    "oa": LicenseType.OPEN_ACCESS,
    "proprietary": LicenseType.PROPRIETARY,
    "all rights reserved": LicenseType.PROPRIETARY,
}


def classify_license(raw: str | None) -> LicenseType:
    """Map a raw license string to a ``LicenseType``.

    Returns ``UNKNOWN`` when the input is ``None``, empty, or not
    recognized.  This is intentionally conservative — unknown licenses
    are excluded from the corpus by default.
    """
    if not raw:
        return LicenseType.UNKNOWN
    key = raw.strip().lower().replace("_", "-")
    return _LICENSE_ALIASES.get(key, LicenseType.UNKNOWN)


def is_open_license(license_type: LicenseType) -> bool:
    """Return ``True`` if the license permits corpus inclusion."""
    return license_type in _OPEN_LICENSES


def is_restricted_license(license_type: LicenseType) -> bool:
    """Return ``True`` if the license forbids corpus inclusion."""
    return license_type in _RESTRICTED_LICENSES


def get_permissions(license_type: LicenseType) -> dict[str, bool]:
    """Return permission flags for a given license type."""
    allowed_for_similarity = is_open_license(license_type) and license_type not in (
        LicenseType.PROPRIETARY,
        LicenseType.UNKNOWN,
        LicenseType.OPEN_ACCESS,
    )
    return {
        "similarity_index_allowed": allowed_for_similarity,
        "training_use_allowed": license_type in (
            LicenseType.CC_BY,
            LicenseType.CC0,
            LicenseType.PUBLIC_DOMAIN,
        ),
        "commercial_use_allowed": license_type in (
            LicenseType.CC_BY,
            LicenseType.CC_BY_SA,
            LicenseType.CC_BY_ND,
            LicenseType.CC0,
            LicenseType.PUBLIC_DOMAIN,
        ),
    }
