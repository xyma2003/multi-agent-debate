"""Literal presentation of saved detector settings, independent of runtime config."""
from __future__ import annotations

from debate.state import DetectorMetadata


def detector_summary(metadata: DetectorMetadata | None) -> str:
    if metadata is None:
        return "Saved detector: unknown — no detector metadata was recorded."
    count = metadata.evaluated_pair_count
    pairs = "pair" if count == 1 else "pairs"
    return f"Saved detector: {metadata.mode.upper()} · {count} evaluated {pairs}."


def detector_details(metadata: DetectorMetadata | None) -> list[tuple[str, str]]:
    """Return plain strings for the UI and Markdown's literal text renderer."""
    if metadata is None:
        return [("Provenance", "Unknown; current settings cannot identify this round's detector.")]
    if metadata.cosine_fast_path_threshold is not None:
        fast_path = repr(metadata.cosine_fast_path_threshold)
    elif metadata.mode == "nli" or metadata.input_field == "reasoning":
        fast_path = "Not used"
    else:
        fast_path = "Not recorded"
    if metadata.nli_contradiction_label_index is not None:
        nli_label = str(metadata.nli_contradiction_label_index)
    elif metadata.mode == "cosine":
        nli_label = "Not applicable"
    else:
        nli_label = "Not recorded"
    details = [
        ("Metadata schema version", str(metadata.schema_version)),
        ("Mode", metadata.mode),
        ("Model ID", metadata.model_id),
        ("Input field", metadata.input_field),
        ("Divergence threshold", repr(metadata.divergence_threshold)),
        ("Cosine fast-path threshold", fast_path),
        ("NLI contradiction label index", nli_label),
        ("Evaluated pair count", str(metadata.evaluated_pair_count)),
        ("Sentinel filtering", "Enabled" if metadata.sentinels_excluded else "Disabled"),
    ]
    if metadata.evaluated_pair_count == 0:
        details.append(("Evaluation", "No pairs were evaluated; no pairwise model inference was run."))
    details.append(("Model identity", "The saved model ID is a loader identifier, not a pinned weight revision."))
    return details
