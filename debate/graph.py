# debate/graph.py
"""
StateGraph assembly for the multi-agent debate system.

Phase 2 topology (loop + fan-out/fan-in):

  START --> initialize --> dispatch_round1 (routing fn)
                              |
             (Send fan-out -- 3 parallel branches)
             |               |               |
        optimist_node  pessimist_node   devil_node
             |               |               |
             +-----------+---+---------------+
                         |
                    collect_round1
                         |
                  divergence_check_node
                         |
              [route_divergence — routing fn]
                /                          \\
    diverged (list[Send])         converged or max_rounds
         /                                  \\
   fan-out: optimist/pessimist/devil     synthesize_stub
         \\                                  |
          collect_round1 (reused)          END
          (loop back to divergence_check)

Import: `from debate.graph import graph` then
        `graph.invoke({"topic": ..., "max_rounds": 3}, config=...)`
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from debate.nodes.agents import devil_node, optimist_node, pessimist_node
from debate.nodes.collect import collect_round1
from debate.nodes.dispatch import dispatch_round1, route_divergence
from debate.nodes.divergence_check import divergence_check_node
from debate.nodes.initialize import initialize_node
from debate.nodes.synthesize import synthesize_stub
from debate.state import DebateState


def build_graph():
    """Build and compile the Phase 2 debate graph."""
    builder = StateGraph(DebateState)

    # --- Register nodes ---
    # NOTE: dispatch_round1 and route_divergence are NOT registered as nodes.
    # They are routing functions passed to add_conditional_edges. Registering
    # them as nodes causes InvalidUpdateError (list[Send] is not a state dict).
    builder.add_node("initialize", initialize_node)
    builder.add_node("optimist_node", optimist_node)
    builder.add_node("pessimist_node", pessimist_node)
    builder.add_node("devil_node", devil_node)
    builder.add_node("collect_round1", collect_round1)
    builder.add_node("divergence_check_node", divergence_check_node)
    builder.add_node("synthesize_stub", synthesize_stub)

    # --- Edges ---

    # START → initialize (linear)
    builder.add_edge(START, "initialize")

    # initialize → parallel fan-out (Round 1)
    # dispatch_round1 returns list[Send]; passed as routing fn, NOT a node.
    builder.add_conditional_edges("initialize", dispatch_round1)

    # Three agent nodes → collect_round1 (fan-in; same node reused for rebuttal rounds)
    builder.add_edge("optimist_node", "collect_round1")
    builder.add_edge("pessimist_node", "collect_round1")
    builder.add_edge("devil_node", "collect_round1")

    # collect_round1 → divergence_check_node (Phase 2 replaces the old → END edge)
    builder.add_edge("collect_round1", "divergence_check_node")

    # divergence_check_node → route_divergence (routing fn)
    # route_divergence returns EITHER list[Send] (rebuttal fan-out)
    # OR the string "synthesize_stub" (termination).
    # LangGraph 1.1.9 handles both return types correctly.
    builder.add_conditional_edges("divergence_check_node", route_divergence)

    # Termination: synthesize_stub → END
    builder.add_edge("synthesize_stub", END)

    # InMemorySaver required for interrupt() support in Phase 4
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph -- import this directly
graph = build_graph()
