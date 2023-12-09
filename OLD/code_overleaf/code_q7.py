
def question_7(graph):
    """
     Pour le graphe H appliquer l'algorithm Bellman-Ford en utilisant un ordre tiré
    aléatoirement (de manière uniforme).
    """
    ordre = Graph.generate_random_order(graph)
    path,dist,nb_iter,state = graph.bellman_ford(0, ordre)
    print("[INFO] Question 7 terminée\n")
    return path,dist,nb_iter,state