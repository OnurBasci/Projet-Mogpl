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
        Coder l'algorithme Bellman-Ford comme ennoncé plus haut. L'algorithme renverra
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

    #Graph.show_graph(g1)
    #Graph.show_graph(g2)
    #Graph.show_graph(g3)
    #Graph.show_graph(H)

    return g1, g2, g3, H

def question_4(graph_1: Graph, graph_2: Graph, graph_3: Graph):
    """
        Etant donn es les 3 graphes G1,G2,G3 (g ́en ́er ́es dans la question 3), appliquer
        l'algorithme Bellman-Ford dans chacun de ces graphes et d ́eterminer l'union de leurs arborescences
        des plus courts chemins que l'on appelera T.
    """
    print(0)
    path_1,_,_ = graph_1.search_bellman_ford(0,None)
    print(1)
    path_2,_,_ = graph_2.search_bellman_ford(0,None)
    print(2)
    path_3,_,_ = graph_3.search_bellman_ford(0,None)
    print(3)
    union_T : Graph = Graph.unifiy_paths(path_1,path_2,path_3, graph_1.list_vertex)

    return union_T

def question_5(union_T):
    """
     Appliquer l’algorithme GloutonFas avec comme entrée T et retourner un ordre <tot.
    """
    print("Question 5: Calcul de l'ordre pour le graphe d'union")
    ordre = union_T.glouton_fas()
    print(f"Question 5 \n {ordre=}")
    return ordre

def question_6(H, ordre):
    """
    Pour le graphe H généré à la question 3 appliquer l’algorithme Bellman-Ford en
    utilisant l’ordre <tot.
    """
    print("Question 6: Application de Bellman Ford à H")
    return H.search_bellman_ford(0, ordre)

def question_7(graph):
    """
     Pour le graphe H appliquer l’algorithm Bellman-Ford en utilisant un ordre tiré
    aléatoirement (de manière uniforme).
    """
    ordre = Graph.generate_random_order(graph)
    return graph.search_bellman_ford(0, ordre)

def question_8(Bellman_H, Bellman_H_random):
    """
     Comparer les résultats (nombre d’itérations) obtenus dans les questions 6 et 7.
    """
    print("Comparaison Bellman avec prétraitement et Bellman avec un ordre aléatoire")
    if not(Bellman_H) or not(Bellman_H_random):
        return
    print(f"Nb itération, Bellman avec prétraitement: {Bellman_H[1]}")
    print(f"Nb itération, Bellman avec un ordre aléatoire: {Bellman_H_random[1]}")

def main():
    my_graph = get_base_graph()
    question_1(my_graph)
    question_2(my_graph)
    g1,g2,g3,H = question_3(my_graph)
    union_T = question_4(g1,g2,g3)
    ordre = question_5(union_T)
    Bellman_H = question_6(H, ordre)
    Bellman_H_random = question_7(H)
    question_8(Bellman_H, Bellman_H_random)



if __name__ == "__main__":
    main()