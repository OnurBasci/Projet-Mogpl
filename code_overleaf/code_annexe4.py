def anlayse_nb_iter_by_nb_edges(nb_train_graph, nb_sommet, nb_repetitions, max_nb_edges, nb_pas = 5):

    moyens_glouton = []
    moyens_random = []

    for i in range(20, max_nb_edges, nb_pas):
        print(f"iteration {i}")
        moyen_glouton, moyen_random, _, _ = calclulate_mean_nb_iter(nb_graphs=nb_train_graph, nb_vertex=nb_sommet, nb_edges=i, nb_repetition=nb_repetitions)
        moyens_glouton.append(moyen_glouton)
        moyens_random.append(moyen_random)

    nb_train_graphs = [i for i in range(20, max_nb_edges, nb_pas)]