"""Single config file for audit thresholds (spec §5).

Traffic-light thresholds are calibrated once on the model spectrum (D5), then
frozen; they are printed on the site's method page. Do not tune per-model.
"""

LIGHT_GREEN = 0.10   # score <= green  -> "green"
LIGHT_YELLOW = 0.30  # score <= yellow -> "yellow", else "red"

K_SAMPLES = 10       # CRN samples per model contrast
SCORE_CAP = 99.9     # scores are capped here for JSON; light computed pre-cap
DIVERGENCE_LIMIT = 1e3   # |state| beyond this marks a model rollout as diverged
AUDIT_SEED = 0       # root seed for the audit grid


def light(score: float) -> str:
    import math
    if not math.isfinite(score):
        return "red"
    if score <= LIGHT_GREEN:
        return "green"
    if score <= LIGHT_YELLOW:
        return "yellow"
    return "red"
