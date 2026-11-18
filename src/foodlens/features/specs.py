"""The base (frame-level) feature table: one row per feature computed from ONE photo (P4.T1).

Every frame feature is computed on the verified light's channel (RED / GREEN / BLUE) at glint
resolution (``vision.glint_long_side``) inside the session ROI ("lit meat"), the shadowed meat
zone or the lit non-meat reference region (``eda/meat_zones.py``). ``schema.py`` expands these
into the per-light triplet columns, the cross-light ratios and the session-relative variants.
Glints are specular reflections of surface moisture, NOT bacteria.

Columns of ``BASE``: (name, group, unit, description, physical meaning).
"""

from __future__ import annotations

GREY = "grey level 0-255"
FRAC = "fraction 0-1"
DENS = "count per 10^4 ROI px"
PX = "px at glint resolution"
UNITLESS = "unitless"

_ROI = "of the lit channel inside the lit session ROI, glint pixels excluded"
_REFL = "diffuse reflectance of the meat surface under this LED (drying, colour change)"
_GLINT_REL = "relative-rule glints (top-hat >= 1.75 x local surface), ROI-edge band dropped"
_GLINT_ABS = "OLD METHOD baseline: glints with the fixed floor glint.abs_min (40 grey levels)"
_WET = "surface moisture: specular highlights from a wet film shrink as the surface dries"

_INTENSITY = [
    ("roi_median", GREY, f"median {_ROI}", _REFL),
    ("roi_p10", GREY, f"10th percentile {_ROI}", _REFL + "; darkest meat areas"),
    ("roi_p25", GREY, f"25th percentile {_ROI}", _REFL),
    ("roi_p75", GREY, f"75th percentile {_ROI}", _REFL),
    ("roi_p90", GREY, f"90th percentile {_ROI}", _REFL + "; brightest diffuse areas"),
    ("roi_iqr", GREY, f"interquartile range {_ROI}", "unevenness of the surface reflectance"),
    ("roi_mean", GREY, f"mean {_ROI}", _REFL + " (mean, not median, on dim channels)"),
    ("roi_std", GREY, f"standard deviation {_ROI}", "unevenness of the surface reflectance"),
    ("roi_sat_frac", FRAC, "share of lit-ROI pixels >= glint.sat_level", _WET),
    ("shd_median", GREY, "median of the shadowed meat zone (glints excluded)", _REFL),
    ("shd_mean", GREY, "mean of the shadowed meat zone (glints excluded)", _REFL),
    ("shd_p90", GREY, "90th percentile of the shadowed meat zone", _REFL),
    ("full_median", GREY, "median over lit + shadowed meat", _REFL),
    ("full_mean", GREY, "mean over lit + shadowed meat", _REFL),
]

_RATIO = [
    ("roi_cv", UNITLESS, "roi_std / roi_mean", "relative unevenness; independent of LED gain"),
    ("roi_iqr_rel", UNITLESS, "roi_iqr / roi_median", "relative spread; independent of gain"),
    (
        "roi_mean_rel_ref",
        UNITLESS,
        "roi_mean / mean of the lit non-meat reference (tray / holder / walls)",
        "meat reflectance relative to its surroundings: cancels LED / camera gain changes",
    ),
    ("shd_rel_roi", UNITLESS, "shd_mean / roi_mean", "shadow vs lit meat (surface shape, gloss)"),
]

_GLINT_KEYS = [
    ("density", DENS, "number of glints per 10^4 ROI px"),
    ("area_frac", FRAC, "glint pixels / ROI pixels"),
    ("size_p50", PX, "median glint area (px)"),
    ("size_p90", PX, "90th percentile glint area (px)"),
    ("size_max", PX, "largest glint area (px)"),
    ("ecc_mean", UNITLESS, "mean glint eccentricity (0 = round droplet, 1 = streak)"),
    ("sat_frac", FRAC, "saturated share of all glint pixels"),
    ("brightness_median", GREY, "median of the glints' mean brightness"),
]

_GLINT = (
    [(f"glint_{k}", u, f"{d}; {_GLINT_REL}", _WET) for k, u, d in _GLINT_KEYS]
    + [
        (
            "glint_brightness_rel",
            UNITLESS,
            f"glint_brightness_median / roi_median; {_GLINT_REL}",
            _WET + "; independent of LED gain",
        )
    ]
    + [(f"glintabs_{k}", u, f"{d}; {_GLINT_ABS}", _WET) for k, u, d in _GLINT_KEYS]
    + [("shd_rglint_density", DENS, "relative-rule glints per 10^4 px of the shadow zone", _WET)]
)

_GLCM = [
    ("glcm_contrast", "squared grey levels (32-level ROI quantisation)", "local contrast"),
    ("glcm_homogeneity", UNITLESS, "closeness of co-occurring grey levels"),
    ("glcm_energy", UNITLESS, "uniformity (angular second moment)"),
    ("glcm_correlation", UNITLESS, "linear dependency of neighbouring grey levels"),
]
_TEXTURE_MEANING = "surface micro-texture (fibre structure, drying film, slime); brightness-free"


def base_rows(lbp_p: int) -> list[tuple[str, str, str, str, str]]:
    """(name, group, unit, description, physical meaning) of every frame feature."""
    rows = [(n, "intensity", u, d, m) for n, u, d, m in _INTENSITY]
    rows += [(n, "ratio", u, d, m) for n, u, d, m in _RATIO]
    rows += [(n, "glint", u, d, m) for n, u, d, m in _GLINT]
    rows += [(n, "texture", u, f"GLCM {d} of the lit ROI", _TEXTURE_MEANING) for n, u, d in _GLCM]
    for i in range(lbp_p + 2):
        desc = f"share of ROI pixels with uniform LBP code {i} (P={lbp_p}; {lbp_p + 1} = other)"
        rows.append((f"lbp_{i}", "texture", FRAC, desc, _TEXTURE_MEANING))
    ent = ("lbp_entropy", "texture", "bits", "entropy of the LBP histogram", _TEXTURE_MEANING)
    rows.append(ent)
    return rows
