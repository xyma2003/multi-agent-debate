"""Load the bundled, explicitly synthetic report without running a debate."""
from pathlib import Path

from pydantic import BaseModel

from debate.state import DebateReport


EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "examples" / "offline_debate.json"


class OfflineExample(BaseModel):
    """Display metadata kept separate from the persisted report schema."""

    title: str
    provenance: str
    report: DebateReport


def load_offline_example() -> OfflineExample:
    """Return a freshly validated example; never call a model or write history.

    File access errors propagate as OSError; invalid text, JSON, or schema
    content raises ValueError. The caller can show a recoverable UI message.
    """
    return OfflineExample.model_validate_json(EXAMPLE_PATH.read_text(encoding="utf-8"))
