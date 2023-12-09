def anlayse_nb_iter_by_nb_trainGraph(nb_vertex, nb_edges, nb_repetitions, max_nb_trainGraph):
    moyens_glouton = []
    moyens_random = []

    for i in range(1, max_nb_trainGraph):
        print(f"iteration {i}")
        moyen_glouton, moyen_random, _, _ = calclulate_mean_nb_iter(i, nb_vertex, nb_edges, nb_repetitions)
        moyens_glouton.append(moyen_glouton)
        moyens_random.append(moyen_random)

    nb_train_graph = [i for i in range(1, max_nb_trainGraph)]