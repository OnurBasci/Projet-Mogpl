@staticmethod
def generate_random_graph(size_graph : int,nb_edges : int) -> 'Graph':
    """
        Fonction qui génère un graph aléatoire d'une taile donnée
    """
    liste_vertex : list = [i for i in range(size_graph)]
    liste_edges = [tuple(random.sample(liste_vertex,2)+[1]) for _ in range(nb_edges)]
    new_graph = Graph(liste_vertex)
    new_graph.add_edges(liste_edges)
    new_graph = Graph.generate_random_weights(new_graph)
    return new_graph