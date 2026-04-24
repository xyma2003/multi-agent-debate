# debate/nodes/synthesize.py
"""
Synthesize stub: Phase 3 placeholder that terminates the debate loop.

The real synthesizer (Phase 3: SYNTH-01 through SYNTH-05) will replace this.
For Phase 2, it records the termination reason so tests can verify routing.

Termination reasons:
  "converged"  — divergence_score dropped below threshold naturally
  "max_rounds" — round_num reached max_rounds regardless of divergence
"""
from debate.state import DebateState


def synthesize_stub(state: DebateState) -> dict:
    """Phase 3 placeholder. Records termination reason and prints a summary."""
    round_num = state.get("round_num", 0)
    divergence_score = state.get("divergence_score", 0.0)
    topic = state.get("topic", "(no topic)")
    round_history = state.get("round_history", [])
    max_rounds = state.get("max_rounds", 3)

    termination = "converged" if divergence_score < 0.25 else "max_rounds"

    print(
        f"\n{'='*60}\n"
        f"DEBATE COMPLETE — {topic!r}\n"
        f"Rounds completed: {round_num}\n"
        f"Final divergence score: {divergence_score:.3f}\n"
        f"Termination reason: {termination}\n"
        f"Total arguments: {sum(len(r.arguments) for r in round_history)}\n"
        f"{'='*60}\n"
    )
    return {"status": termination}
