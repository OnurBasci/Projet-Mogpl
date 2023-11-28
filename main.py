import numpy as np
from graph import Graph


def get_base_graph():
    my_graph = Graph([0,1,2,3,4,5,6,7])
    liste_arcs = [(0,1,1),(0,2,1),(1,2,1),(2,3,1),(3,6,1),(3,4,1),(3,5,1),(4,6,1),(5,4,1),(5,7,1),(6,0,1),(7,2,1),(7,1,1)]
    my_graph.add_edges(liste_arcs)
    return my_graph

def question_1(graph : Graph) -> None:
    graph.search_bellman_ford(0,None)
    graph.show_bellmanford_result()

def question_2(graph : Graph) -> list:
    ordre = graph.glouton_fas()
    print("=========================\n")	
    print(f"[INFO] Ordre obtenu : {ordre}")
    return ordre

def question_3(graph : Graph):
    G1 = Graph.generate_random_weights(graph)
    G2 = Graph.generate_random_weights(graph)
    G3 = Graph.generate_random_weights(graph)
    H = Graph.generate_random_weights(graph)

    print(f"{G1.graph=}")
    print(f"{G2.graph=}")
    print(f"{G3.graph=}")
    print(f"{H.graph=}")
    return G1, G2, G3, H


def question_4(graph_1: Graph, graph_2: Graph, graph_3: Graph):
    print(0)
    path_1,_,_ = graph_1.search_bellman_ford(0,None)
    print(1)
    path_2,_,_ = graph_2.search_bellman_ford(0,None)
    print(2)
    path_3,_,_ = graph_3.search_bellman_ford(0,None)
    print(3)

    union_T : Graph = Graph.unifiy_paths(path_1,path_2,path_3, graph_1.list_vertex)
    print(f"{path_1=}")
    print(f"{path_2=}")
    print(f"{path_3=}")
    print(f"{union_T.graph=}")
    ordre = union_T.glouton_fas()
    print(f"{ordre=}")

def main():
    my_graph = get_base_graph()
    question_1(my_graph)
    question_2(my_graph)
    G1, G2, G3, H = question_3(my_graph)
    question_4(G1, G2, G3)

if __name__ == "__main__":
    main()