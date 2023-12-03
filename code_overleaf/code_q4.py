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