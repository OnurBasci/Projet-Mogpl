from graph import Graph
from analyse import anlayse_nb_iter_by_nb_trainGraph, calclulate_mean_nb_iter

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
    graph.bellman_ford(0,None)
    graph.show_bellmanford_result()
    print("[INFO] Question 1 terminée\n")

def question_2(graph : Graph) -> list:
    """
        Coder l'algorithme GloutonFas qui prend comme entr ́ee un graphe et renverra un
        arrangement lin ́eaire de ses sommets.
    """
    ordre = graph.glouton_fas()
    print("=========================\n")	
    print("[INFO] Question 2 terminée\n")
    print(f"[INFO] Ordre obtenu : {ordre}")
    return ordre

def question_3(graph : Graph) -> (Graph,Graph,Graph):
    """
        A partir de G construire 3 graphes ponder ́es G1,G2,G3 ainsi que le graphe test H.
    """
    g1, g2, g3, H = Graph.generate_graphs_with_random_weights(graph,4)

    print("[INFO] Question 3 terminée\n")

    return g1, g2, g3, H

def question_4(graph_1: Graph, graph_2: Graph, graph_3: Graph):
    """
        Etant donnes les 3 graphes G1,G2,G3 (g ́en ́er ́es dans la question 3), appliquer
        l'algorithme Bellman-Ford dans chacun de ces graphes et d ́eterminer l'union de leurs arborescences
        des plus courts chemins que l'on appelera T.
    """
    path_1,_,_,_ = graph_1.bellman_ford(0,None)
    path_2,_,_,_ = graph_2.bellman_ford(0,None)
    path_3,_,_,_ = graph_3.bellman_ford(0,None)


    union_T : Graph = Graph.unifiy_paths([path_1,path_2,path_3], graph_1.list_vertex)
    print("[INFO] Question 4 terminée\n")
    return union_T  

def question_5(union_T):
    """
     Appliquer l'algorithme GloutonFas avec comme entrée T et retourner un ordre <tot.
    """
    print("Question 5: Calcul de l'ordre pour le graphe d'union")
    ordre = union_T.glouton_fas()
    print(f"Question 5 \n {ordre=}")
    return ordre

def question_6(H, ordre):
    """
    Pour le graphe H généré à la question 3 appliquer l'algorithme Bellman-Ford en
    utilisant l'ordre <tot.
    """
    print("Question 6: Application de Bellman Ford à H")
    return H.bellman_ford(0,ordre)

def question_7(graph):
    """
     Pour le graphe H appliquer l'algorithm Bellman-Ford en utilisant un ordre tiré
    aléatoirement (de manière uniforme).
    """
    ordre = Graph.generate_random_order(graph)
    path,dist,nb_iter,state = graph.bellman_ford(0, ordre)
    print("[INFO] Question 7 terminée\n")
    return path,dist,nb_iter,state

def question_8(Bellman_H, Bellman_H_random):
    """
     Comparer les résultats (nombre d'itérations) obtenus dans les questions 6 et 7.
    """
    print("Comparaison Bellman avec prétraitement et Bellman avec un ordre aléatoire")
    if not(Bellman_H) or not(Bellman_H_random):
        return
    print(f"Nb itération, Bellman avec prétraitement: {Bellman_H[2]}")
    print(f"Nb itération, Bellman avec un ordre aléatoire: {Bellman_H_random[2]}")
    print("[INFO] Question 8 terminée\n")

def question_9():
    """
        Génerer des instances al ́eatoires pour tester la m ́ethode avec pr ́etraitement par rap-
        port `a l'application de l'algorithme de Bellman-Ford bas ́ee simplement sur un ordre tir e de manièere
        aléatoire
    """
    nb_graphs = 10
    nb_vertex = 100
    nb_edges = nb_vertex*2
    Graph.generate_compare_graph(nb_vertex,nb_edges,nb_graphs)
    print("[INFO] Question 9 terminée\n")

def question_10():
    """
    Comment le nombre de graphes que l’on utilise pour apprendre l’ordre influence le
    nombre d’it´erations ?
    """
    print("Question 10")
    anlayse_nb_iter_by_nb_trainGraph(nb_vertex=100, nb_edges=400, nb_repetitions=20, max_nb_trainGraph=10)
    print("[INFO] Question 10 terminée\n")

def question_11():
    """
     Soit la famille d’instances suivantes : un graphe par niveau avec 4 sommets par
    niveau et 2500 niveaux o`u les sommets du niveau j précédent tous les sommets du niveau j + 1
    et où pour chaque arc on a un poids tiré de manière uniforme et aléatoire parmi les entiers dans
    l’intervalle [−10, 10]. Est-ce que la méthode avec prétraitement est adéquate ? Justifier votre
    réponse
    """
    level_graphe = Graph.generate_level_graph(2500)
    calclulate_mean_nb_iter(graphe=level_graphe, nb_repetition=1)
    print("[INFO] Question 11 terminée\n")




def main():
    """
        Fonction principale qui regroupe tous les appels aux fonctions des questions
    """
    # Graph de base
    my_graph = get_base_graph()
    # Question 1
    #question_1(my_graph)
    # Question 2
    question_2(my_graph)
    # Question 3
    g1,g2,g3,H = question_3(my_graph)
    # Question 4
    union_T = question_4(g1,g2,g3)
    # Question 5
    ordre = question_5(union_T)
    # Question 6
    Bellman_H = question_6(H, ordre)
    # Question 7
    Bellman_H_random = question_7(H)
    # Question 8
    question_8(Bellman_H, Bellman_H_random)
    # Question 9
    question_9()
    # Question 10
    question_10()
    #question 11
    question_11()

if __name__ == "__main__":
    main()