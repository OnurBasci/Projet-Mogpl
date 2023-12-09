def anlayse_nb_iter_by_nb_sommets(nb_train_graph, nb_edges, nb_repetitions, max_nb_sommets, nb_pas = 10):
    moyens_glouton = []
    moyens_random = []

    for i in range(10, max_nb_sommets, nb_pas):
        print(f"iteration {i}")
        moyen_glouton, moyen_random, _, _ = calclulate_mean_nb_iter(nb_train_graph, i, nb_edges, nb_repetitions)
        moyens_glouton.append(moyen_glouton)
        moyens_random.append(moyen_random)

    nb_sommets = [i for i in range(10, max_nb_sommets, nb_pas)]