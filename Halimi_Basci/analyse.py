import matplotlib.pyplot as plt
from graph import Graph


#QUESTION 9
def anlayse_nb_iter_by_random_graph(nb_graphs=10, nb_vertex=100, nb_edges=400):
    """
        La fonction calculant le nombre d'itération de Bellman-Ford, avec un ordre optimale, et aléatoire en fonction de différents graphes
        :param nb_graphs: Nombre de graphes
        :param nb_vertex: Nombre de sommets
        :param nb_edges: Nombre d'arêtes
        :return: None
    """

    nb_edges = nb_vertex*3
    liste_nb_iter_glouton = []
    liste_nb_iter_random = []
    for i in range(nb_graphs):
        print(f"Graphe générés {i+1}/{nb_graphs}")
        nb_iter_glouton,nb_iter_random = Graph.generate_compare_graph(nb_vertex,nb_edges,nb_graph_to_generate=10)
        liste_nb_iter_glouton.append(nb_iter_glouton)
        liste_nb_iter_random.append(nb_iter_random)
        
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(liste_nb_iter_glouton, label="Ordre glouton_fas")
    plt.plot(liste_nb_iter_random, label="Ordre aléatoire")
    plt.legend()
    plt.title("Nombre d'itérations en fonction de différents graphes")
    plt.xlabel("Nombre de graphes")
    plt.ylabel("Nombre d'itérations")
    plt.show()
    mean_glouton = sum(liste_nb_iter_glouton)/len(liste_nb_iter_glouton)
    mean_random = sum(liste_nb_iter_random)/len(liste_nb_iter_random)
    print(f"moyenne glouton {mean_glouton}")
    print(f"moyenne random {mean_random}")
    print(mean_random/mean_glouton)


#QUESTION 10
def calclulate_mean_nb_iter(nb_graphs=7, nb_vertex=100, nb_edges=400, nb_repetition=20, graphe = None):
    """
    La fonction calculant les moyens de nombres d'itération obtenus à partir de l'algorithme Bellman-Ford avec l'ordre
    glouton et l'ordre aléatoire.
    :param nb_graphs: Nombre de graphe d'entrâinement
    :param nb_vertex: Nombre de sommets
    :param nb_edges: nombre des arêtes
    :param nb_repetition: nombre de fois qu'on calcule la moyenne
    :param graphe: Un graphe ou on fait la comparaison
    :return:
    """
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

    print(f"moyenne glouton {moyen_glouton/counter}")
    print(f"moyenne random {moyen_random/counter}")
    print(f"gloutons {gloutons}")
    print(f"randoms {randoms}")

    return moyen_glouton/counter, moyen_random/counter, gloutons, randoms



def anlayse_nb_iter_by_nb_trainGraph(nb_vertex, nb_edges, nb_repetitions, max_nb_trainGraph):
    """
    Faire une analyse de la peformence en fonction de nombre de graphe d'entraînement
    :param nb_vertex: nombre de sommets
    :param nb_edges: nombre d'arêtes
    :param nb_repetitions: nombre de fois qu'on calcule la moyenne par itérations
    :param max_nb_trainGraph: Maximum de nombre de graphe d'entrâinement
    :return: None
    """
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

    plt.style.use("ggplot")

    fig, ax = plt.subplots()

    # Plot both functions on the same subplot
    ax.plot(nb_train_graph, moyens_glouton, label="Moyen glouton_fas", color='blue')
    ax.plot(nb_train_graph, moyens_random, label="Moyen aléatoire", color='red')

    # Set title and labels
    ax.set_title("Nombre moyen d'itérations en fonction du nombre de graphes d'entraînement.")
    ax.set_xlabel("Nombre de graphe d'entraînement")
    ax.set_ylabel("Nombre d'itérations moyens")

    # Display a legend
    ax.legend()

    plt.savefig(f"Comparaison_en_fonction_de_nb_graphe_{max_nb_trainGraph}_{nb_vertex}_{nb_edges}_{nb_repetitions}.png")

    # Show the plot
    plt.show()


def anlayse_nb_iter_by_nb_sommets(nb_train_graph, nb_edges, nb_repetitions, max_nb_sommets, nb_pas = 10):
    """
    Faire une analyse de la peformence en fonction de nombre de graphe d'entraînement
    :param nb_train_graph: Nombre de graphe d'entraînement
    :param nb_edges: Nombre d'arêtes
    :param nb_repetitions: nombre de fois qu'on calcule la moyenne par itérations
    :param max_nb_sommets: Maximum de nombre de sommet
    :param nb_pas: la difference entre deux nb sommets
    :return: None
    """
    moyens_glouton = []
    moyens_random = []

    for i in range(10, max_nb_sommets, nb_pas):
        print(f"iteration {i}")
        moyen_glouton, moyen_random, _, _ = calclulate_mean_nb_iter(nb_train_graph, i, nb_edges, nb_repetitions)
        moyens_glouton.append(moyen_glouton)
        moyens_random.append(moyen_random)

    nb_sommets = [i for i in range(10, max_nb_sommets, nb_pas)]

    print(moyens_random)
    print(moyens_glouton)
    print(nb_sommets)

    plt.style.use("ggplot")

    fig, ax = plt.subplots()

    # Plot both functions on the same subplot
    ax.plot(nb_sommets, moyens_glouton, label="Ordre glouton_fas", color='blue')
    ax.plot(nb_sommets, moyens_random, label="Odre aléatoire", color='red')

    # Set title and labels
    ax.set_title("Nombre moyen d'itérations en fonction de nombre de sommets")
    ax.set_xlabel("Nombre de sommets")
    ax.set_ylabel("Nombre moyen d'itérations")

    # Display a legend
    ax.legend()

    plt.savefig(f"Comparaison_en_fonction_de_nb_sommets_{max_nb_sommets}_{nb_train_graph}_{nb_edges}_{nb_repetitions}.png")

    # Show the plot
    plt.show()

def anlayse_nb_iter_by_nb_edges(nb_train_graph, nb_sommet, nb_repetitions, max_nb_edges, nb_pas = 5):
    """
    :param nb_train_graph: Nombre de graphe d'entraînement
    :param nb_sommet: Nombre de sommet
    :param nb_repetitions: nombre de fois qu'on calcule la moyenne par itérations
    :param max_nb_edges: Maximum de nombre d'arêtes
    :param nb_pas: la difference entre deux nombre d'arêtes.
    :return:
    """
    moyens_glouton = []
    moyens_random = []

    for i in range(100, max_nb_edges, nb_pas):
        print(f"iteration {i}")
        moyen_glouton, moyen_random, _, _ = calclulate_mean_nb_iter(nb_graphs=nb_train_graph, nb_vertex=nb_sommet, nb_edges=i, nb_repetition=nb_repetitions)
        moyens_glouton.append(moyen_glouton)
        moyens_random.append(moyen_random)

    nb_train_graphs = [i for i in range(100, max_nb_edges, nb_pas)]

    print(moyens_random)
    print(moyens_glouton)
    print(nb_train_graphs)

    plt.style.use("ggplot")

    plt.style.use("ggplot")
    fig, ax = plt.subplots()

    # Plot both functions on the same subplot
    ax.plot(nb_train_graphs, moyens_glouton, label="Ordre glouton_fas", color='blue')
    ax.plot(nb_train_graphs, moyens_random, label="Ordre aléatoire", color='red')

    # Set title and labels
    ax.set_title("Nombre moyen d'itérations en fonction du nombre d'arêtes")
    ax.set_xlabel("Nombre d'arêtes")
    ax.set_ylabel("Nombre moyen d'itérations")

    # Display a legend
    ax.legend()

    plt.savefig(f"Comparaison_en_fonction_de_nb_edges_{max_nb_edges}_{nb_train_graph}_{nb_sommet}_{nb_repetitions}")

    # Show the plot
    plt.show()


