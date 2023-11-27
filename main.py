from graph import Graph

def get_base_graph() -> Graph :
    """ 
    Fonction qui retourne un graphe de base pour les tests
    :return: Graph
    """
    my_graph = Graph([0,1,2,3,4,5,6,7])
    liste_arcs = [(0,1,1),(0,2,1),(1,2,1),(2,3,1),(3,6,1),(3,4,1),(3,5,1),(4,6,1),(5,4,1),(5,7,1),(6,0,1),(7,2,1),(7,1,1)]
    my_graph.add_edges(liste_arcs)
    return my_graph

def question_1(graph : Graph) -> None:
    """
        Coder l'algorithme Bellman-Ford comme ennonc ́e plus haut. L'algorithme renverra
        l'arborescence des plus courts chemins ainsi que le nombre d'it ́erations qui ont  ́et ́e n ́ecessaires avant
        que l'algorithme converge
    """
    graph.search_bellman_ford(0,None)
    graph.show_bellmanford_result()

def question_2(graph : Graph) -> list:
    """
        Coder l'algorithme GloutonFas qui prend comme entr ́ee un graphe et renverra un
        arrangement lin ́eaire de ses sommets.
    """
    ordre = graph.glouton_fas()
    print("=========================\n")	
    print(f"[INFO] Ordre obtenu : {ordre}")
    return ordre

def question_3(graph : Graph) -> (Graph,Graph,Graph):
    """
        A partir de G construire 3 graphes ponder ́es G1,G2,G3 ainsi que le graphe test H.
    """
    g1 = Graph.generate_random_weights(graph)
    g2 = Graph.generate_random_weights(graph)
    g3 = Graph.generate_random_weights(graph)
    H = Graph.generate_random_weights(graph)

    Graph.show_graph(g1)
    Graph.show_graph(g2)
    Graph.show_graph(g3)
    Graph.show_graph(H)

    return g1,g2,g3

def question_4(graph_1: Graph, graph_2: Graph, graph_3: Graph):
    """
        Etant donn es les 3 graphes G1,G2,G3 (g ́en ́er ́es dans la question 3), appliquer
        l'algorithme Bellman-Ford dans chacun de ces graphes et d ́eterminer l'union de leurs arborescences
        des plus courts chemins que l'on appelera T.
    """
    path_1,_,_ = graph_1.search_bellman_ford(0,None)
    path_2,_,_ = graph_2.search_bellman_ford(0,None)
    path_3,_,_ = graph_3.search_bellman_ford(0,None)

    union_T : Graph = Graph.unifiy_paths(path_1,path_2,path_3)
    ordre = union_T.glouton_fas()


def main():
    my_graph = get_base_graph()
    question_1(my_graph)
    question_2(my_graph)
    g1,g2,g3 = question_3(my_graph)
    question_4(g1,g2,g3)

if __name__ == "__main__":
    main()