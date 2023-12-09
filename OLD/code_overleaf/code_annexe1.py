def anlayse_nb_iter_by_random_graph(nb_graphs=10, nb_vertex=100, nb_edges=400):
    nb_edges = nb_vertex*2
    liste_nb_iter_glouton = []
    liste_nb_iter_random = []
    for _ in range(nb_graphs):
        nb_iter_glouton,nb_iter_random = Graph.generate_compare_graph(nb_vertex,nb_edges,nb_graph_to_generate=10)
        liste_nb_iter_glouton.append(nb_iter_glouton)
        liste_nb_iter_random.append(nb_iter_random)

    mean_glouton = sum(liste_nb_iter_glouton)/len(liste_nb_iter_glouton)
    mean_random = sum(liste_nb_iter_random)/len(liste_nb_iter_random)

    return mean_glouton, mean_random