from langgraph.graph import END, START, StateGraph

from src.graph.nodes import RAGNodes, route_after_scope
from src.graph.state import RAGState


def build_rag_graph(nodes: RAGNodes):
    """
    START -> retrieve -> check_scope -+-> generate_from_pdf -> END
                ^                     +-> rewrite_query --+
                +-------------------------------------------+
                                      +-> [web_search ->] generate_general -> END

    The web_search node exists only when a search client is configured (TAVILY_API_KEY).
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("check_scope", nodes.check_scope)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate_from_pdf", nodes.generate_from_pdf)
    graph.add_node("generate_general", nodes.generate_general)
    web = nodes.web_search_client is not None
    if web:
        graph.add_node("web_search", nodes.web_search)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "check_scope")
    graph.add_conditional_edges(
        "check_scope",
        route_after_scope,
        {
            "generate_from_pdf": "generate_from_pdf",
            "rewrite_query": "rewrite_query",
            "generate_general": "web_search" if web else "generate_general",
        },
    )
    if web:
        graph.add_edge("web_search", "generate_general")
    graph.add_edge("rewrite_query", "retrieve")  # the corrective loop
    graph.add_edge("generate_from_pdf", END)
    graph.add_edge("generate_general", END)

    return graph.compile()
