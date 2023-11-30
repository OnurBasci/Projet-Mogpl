import matplotlib.pyplot as plt
from graph import Graph

#QUESTION 10
def calclulate_mean_nb_iter(nb_graphs=7, nb_vertex=100, nb_edges=400, nb_repetition=20, graphe = None):
    moyen_glouton = 0
    moyen_random = 0
    gloutons = []
    randoms = []
    counter = 0
    for i in range(nb_repetition):
        print(f"iteration : {i}")
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

    print(f"moyenne glouton {moyen_glouton/counter}")
    print(f"moyenne random {moyen_random/counter}")
    print(f"gloutons {gloutons}")
    print(f"randoms {randoms}")

    return moyen_glouton/counter, moyen_random/counter, gloutons, randoms



def anlayse_nb_iter_by_nb_trainGraph(nb_vertex, nb_edges, nb_repetitions, max_nb_trainGraph):
    moyens_glouton = []
    moyens_random = []

    for i in range(1, max_nb_trainGraph):
        moyen_glouton, moyen_random, _, _ = calclulate_mean_nb_iter(i, nb_vertex, nb_edges, nb_repetitions)
        moyens_glouton.append(moyen_glouton)
        moyens_random.append(moyen_random)

    nb_train_graph = [i for i in range(1, max_nb_trainGraph)]

    print(moyens_random)
    print(moyens_glouton)
    print(nb_train_graph)

    fig, ax = plt.subplots()

    # Plot both functions on the same subplot
    ax.plot(nb_train_graph, moyens_glouton, label="Moyen Glouton", color='blue')
    ax.plot(nb_train_graph, moyens_random, label="Moyen random", color='red')

    # Set title and labels
    ax.set_title("Moyens nombre d'itérations en fonction de Graphe d'entraînement")
    ax.set_xlabel("Npmbre de graphe d'entraînement")
    ax.set_ylabel("Moyen de nombre d'itérations")

    # Display a legend
    ax.legend()

    # Show the plot
    plt.show()
