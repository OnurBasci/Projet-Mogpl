@staticmethod
def generate_random_order(graph):
    """
    Fonction qui génère un ordre aléatoire pour un graphe donnée
    """
    nb_elem = len(graph.list_vertex)
    return random.sample(range(nb_elem), nb_elem)