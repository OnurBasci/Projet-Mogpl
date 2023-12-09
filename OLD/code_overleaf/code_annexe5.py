def calclulate_mean_nb_iter(nb_graphs=7, nb_vertex=100, nb_edges=400, nb_repetition=20, graphe = None):
    moyen_glouton = 0
    moyen_random = 0
    gloutons = []
    randoms = []
    counter = 0
    for i in range(nb_repetition):
        if graphe is None:
            nb_iter_glouton, nb_iter_random = Graph.generate_compare_graph(nb_vertex, nb_edges, nb_graphs)
        else:
            nb_iter_glouton, nb_iter_random = Graph.compare_graph(graphe, nb_graphs)
        if nb_iter_random is None or nb_iter_glouton is None:
            continue
        gloutons.append(nb_iter_glouton)
        randoms.append(nb_iter_random)
        moyen_glouton += nb_iter_glouton
        moyen_random += nb_iter_random
        counter += 1

    return moyen_glouton/counter, moyen_random/counter, gloutons, randoms