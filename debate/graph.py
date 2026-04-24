# debate/graph.py
"""
StateGraph assembly for the multi-agent debate system.

Phase 1 topology (linear + fan-out/fan-in):

  START --> initialize --> dispatch_round1
                              |
             (Send fan-out -- 3 parallel branches)
             |               |               |
        optimist_node  pessimist_node   devil_node
             |               |               |
             +-----------+---+---------------+
                         |
                    collect_round1 --> END

Phase 2 will replace the collect_round1 --> END edge with conditional routing
to a divergence detector and rebuttal loop.

Import: `from debate.graph import graph` then
        `graph.invoke({"topic": ..., "max_rounds": 3}, config=...)`
"""
from langgraph.checkpoint.memory import InMemorySaver  # NOT MemorySaver -- deprecated alias
from langgraph.graph import END, START, StateGraph

from debate.nodes.agents import devil_node, optimist_node, pessimist_node
from debate.nodes.collect import collect_round1
from debate.nodes.dispatch import dispatch_round1
from debate.nodes.initialize import initialize_node
from debate.state import DebateState


def build_graph():
    """Build and compile the Phase 1 debate graph."""
    builder = StateGraph(DebateState)

    # Register all nodes
    builder.add_node("initialize", initialize_node)
    builder.add_node("dispatch_round1", dispatch_round1)
    builder.add_node("optimist_node", optimist_node)
    builder.add_node("pessimist_node", pessimist_node)
    builder.add_node("devil_node", devil_node)
    builder.add_node("collect_round1", collect_round1)

    # Linear edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "dispatch_round1")

    # Fan-out: dispatch_round1 returns list[Send] -- identity lambda passes it through
    # DO NOT use add_edge here -- that creates a single edge, not a parallel fan-out
    builder.add_conditional_edges("dispatch_round1", lambda s: s)

    # Fan-in: all three agent nodes converge to collect_round1
    builder.add_edge("optimist_node", "collect_round1")
    builder.add_edge("pessimist_node", "collect_round1")
    builder.add_edge("devil_node", "collect_round1")

    # Phase 1 terminal edge -- Phase 2 will replace this
    builder.add_edge("collect_round1", END)

    # InMemorySaver required for interrupt() support in Phase 4
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Module-level compiled graph -- import this directly
graph = build_graph()
