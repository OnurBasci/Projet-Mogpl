@staticmethod
def generate_random_weights(graph : 'Graph'):
    """
    Fonction qui génère des poids aléatoires pour un graphe donnée
    :param graph: Un graphe orienté sans poids
    :return: Creation d'un nouveau graphe avec des poids aléatoirement générés
    """

    # Génération des poids aléatoires entre -10 et 10
    random_graph = Graph(graph.list_vertex)
    list_edges = []
    for vertex in graph.list_vertex:
        for neighbor, _ in graph.graph[vertex]:
            random_weight = random.randint(-10, 10)
            list_edges.append((vertex, neighbor, random_weight))
    random_graph.add_edges(list_edges)

    # Vérification qu'il n'y a pas de cycle négatif
    paths, _, _, contains_negatif_cycle = random_graph.bellman_ford(0, None)
    path = paths[np.argmax(np.array([len(path) for path in paths]))]
    # Si il y a un cycle négatif, on le supprime
    while contains_negatif_cycle :
        # On récupère le cycle négatif
        negative_cycle = Graph._find_negative_cycle(path)
        # On convertit les poids négatifs en positif
        for i in range(len(negative_cycle)-1):
            for u,v,w in list_edges:
                if u == negative_cycle[i] and v == negative_cycle[i+1]:
                    list_edges.remove((u,v,w))
                    list_edges.append((u,v,abs(w)))

        # On fait la même chose pour le dernier arc
        for u, v, w in list_edges:
            if u == negative_cycle[-1] and v == negative_cycle[0]:
                list_edges.remove((u, v, w))
                list_edges.append((u, v, abs(w)))
        random_graph.add_edges(list_edges)
        
        paths, _, _, contains_negatif_cycle = random_graph.bellman_ford(0, None)
        path = paths[np.argmax(np.array([len(path) for path in paths]))]
    return random_graph
