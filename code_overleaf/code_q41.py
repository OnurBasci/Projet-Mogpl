@staticmethod
def unifiy_paths(paths : list,list_vertex : list) -> 'Graph':
    """
    Fonction qui unifie les chemins pour créer un graph
    :param paths: chemins des graphes
    :param list_vertex: liste des sommets
    :return: un graph
    """
    # Initialisation des variables
    list_edges : list = []
    added_edges : dict = {}
    # Fonction qui ajoute un arc au graph si il n'existe pas déjà
    def add_edge_from_path(path):
        if len(path) <= 1:
            return
        for i in range(0,len(path) - 1):
            vertex_1,vertex_2 = path[i],path[i+1]
            if added_edges.get((vertex_1,vertex_2),None) is not None:
                continue
            list_edges.append((path[i],path[i+1],1))
            added_edges[(vertex_1,vertex_2)] = True

    # Ajout des arcs pour chaque chemin
    for p_x in zip(*paths):
        for p in p_x :
            add_edge_from_path(p)
    # Création du graph
    new_graph = Graph(list_vertex)
    new_graph.add_edges(list_edges)
    # Retourne le graph
    return new_graph