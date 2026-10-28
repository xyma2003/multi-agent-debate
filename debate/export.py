"""Pure, portable exports of the report currently shown in the app."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import string
from typing import Literal

from debate.state import Concession, DebateReport
from debate.detector_display import detector_details, detector_summary


@dataclass(frozen=True)
class ReportExports:
    markdown: str
    json: str
    markdown_filename: str
    json_filename: str


_SOURCE_DESCRIPTIONS = {
    "history": "Saved debate report opened from local history.",
    "live": "Report from the completed debate currently shown in this session.",
    "sample": (
        "Fixed synthetic example; no model execution. "
        "This example is not model-quality evidence."
    ),
}

_INTERPRETATION = {
    "saved_values": "Saved report values are preserved, not recomputed during export.",
    "confidence_score": (
        "The legacy confidence_score field stores the Debate Agreement Index on a "
        "0–1 scale. Its 0–100 display is the saved value multiplied by 100. "
        "It is a heuristic, not correctness probability, evidence quality, or proof of consensus."
    ),
    "agent_confidence": (
        "Each role's confidence is self-reported and uncalibrated; "
        "it is not an input to the report's agreement index."
    ),
    "divergence": (
        "Detector metadata, when present, records each round's saved settings. "
        "Rounds without metadata have an unknown detector and configuration. "
        "Do not infer them from the app's current settings. NLI and cosine divergence "
        "scores are not directly comparable, and zero does not prove agreement."
    ),
    "round_numbering": (
        "Stored round_num values are 0-indexed. Markdown also shows their 1-indexed "
        "display numbers; the stored values are preserved."
    ),
    "stop_reason": (
        "convergence_status records why the debate loop stopped. In particular, "
        "converged means the detector found no divergent agent pair; it does not "
        "verify factual correctness or establish real consensus."
    ),
}


def _text_block(value: str) -> str:
    """Quote literal text while allowing long prose to wrap in Markdown viewers."""
    entities = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    quoted = []
    for index, line in enumerate(lines):
        # Encode indentation so arbitrary report text cannot start a code block.
        indent = len(line) - len(line.lstrip(" \t"))
        prefix = "".join("&#32;" if char == " " else "&#9;" for char in line[:indent])
        text = "".join(
            entities.get(char, "\\" + char if char in string.punctuation else char)
            for char in line[indent:]
        )
        # Every logical line stays inside the quote, including blank lines.
        hard_break = "  " if line and index < len(lines) - 1 else ""
        quoted.append(">" + (" " + prefix + text + hard_break if line else ""))
    return "\n".join(quoted)


def _field(label: str, value: str) -> str:
    return f"**{label}**\n\n{_text_block(value)}"


def _concessions(items: list[Concession], heading_level: int) -> list[str]:
    if not items:
        return ["None saved."]
    sections = []
    for index, item in enumerate(items, 1):
        sections.extend([
            f"{'#' * heading_level} Concession {index}",
            _field("Conceded point", item.conceded_point),
            _field("Triggered by role", item.triggered_by_agent),
            _field("Triggered by claim", item.triggered_by_claim),
            _field("Rationale", item.rationale),
        ])
    return sections


def _filename_stem(debate_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", debate_id).strip("-_")
    readable = readable[:48].rstrip("-_") or "report"
    suffix = hashlib.sha256(debate_id.encode("utf-8")).hexdigest()[:12]
    return f"debate-{readable}-{suffix}"


def build_report_exports(
    report: DebateReport,
    *,
    source: Literal["history", "live", "sample"],
    provenance: str | None = None,
) -> ReportExports:
    """Serialize one report without reads, writes, model calls, or input mutation."""
    if source not in _SOURCE_DESCRIPTIONS:
        raise ValueError("source must be history, live, or sample")
    source_info = {"kind": source, "description": _SOURCE_DESCRIPTIONS[source]}
    if provenance is not None:
        source_info["provenance"] = provenance
    payload = {
        "schema_version": 1,
        "source": source_info,
        "interpretation": dict(_INTERPRETATION),
        "report": report.model_dump(mode="json"),
    }

    sections = [
        "# Debate report",
        "## Source",
        _SOURCE_DESCRIPTIONS[source],
        f"Source kind: `{source}`.",
    ]
    if provenance is not None:
        sections.append(_field("Provenance", provenance))
    sections.extend([
        "## Report details",
        _field("Debate ID", report.debate_id),
        _field("Created at (original timezone)", report.created_at.isoformat()),
        "## Topic",
        _text_block(report.topic),
        "## How to read this report",
        *(_INTERPRETATION.values()),
        "## Outcome",
        f"Saved `confidence_score` (0–1): **{report.confidence_score!r}**.",
        f"Debate Agreement Index (0–100 display): **{report.confidence_score * 100:.1f} / 100**.",
        _field("Stop reason (convergence_status)", report.convergence_status),
        "### Verdict",
        _text_block(report.verdict),
        "### Consensus points",
    ])
    if report.consensus_points:
        for index, point in enumerate(report.consensus_points, 1):
            sections.extend([f"#### Consensus point {index}", _text_block(point)])
    else:
        sections.append("None saved.")
    sections.append("### Disputed points")
    if report.disputed_points:
        for index, point in enumerate(report.disputed_points, 1):
            sections.extend([f"#### Disputed point {index}", _field("Topic", point.topic)])
            for role, position in point.agent_positions.items():
                sections.extend([_field("Role", role), _field("Position", position)])
    else:
        sections.append("None saved.")

    sections.append("## Reasoning trace")
    if not report.reasoning_trace:
        sections.append("No rounds were saved.")
    for record in report.reasoning_trace:
        sections.extend([
            f"### Round {record.round_num + 1} (stored round_num: {record.round_num})",
            f"Saved divergence score: **{record.divergence_score!r}**.",
            detector_summary(record.detector_metadata),
            *(_field(label, value) for label, value in detector_details(record.detector_metadata)),
        ])
        if not record.arguments:
            sections.append("No arguments were saved for this round.")
        for index, argument in enumerate(record.arguments, 1):
            sections.extend([
                f"#### Argument {index}",
                _field("Role", argument.agent_role),
                f"Argument round: **{argument.round_num + 1}** "
                f"(stored `round_num`: **{argument.round_num}**).",
                _field("Position", argument.position),
                _field("Reasoning", argument.reasoning),
                f"Saved self-reported confidence (0–1, uncalibrated): **{argument.confidence!r}**.",
                f"Fallback sentinel (`is_sentinel`): **{str(argument.is_sentinel).lower()}**.",
                "##### Key claims",
            ])
            for claim_index, claim in enumerate(argument.key_claims, 1):
                sections.append(_field(f"Claim {claim_index}", claim))
            sections.extend(["##### Concessions in this argument", *_concessions(argument.concessions, 6)])

    sections.extend(["## Global concession log", *_concessions(report.concession_log, 3)])
    stem = _filename_stem(report.debate_id)
    return ReportExports(
        markdown="\n\n".join(sections) + "\n",
        json=json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        markdown_filename=f"{stem}.md",
        json_filename=f"{stem}.json",
    )
