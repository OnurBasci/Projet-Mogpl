def bellman_ford(self, source_vertex, vertex_order=None):
    self.nb_iter = 0
    """
        Fonction qui calcule le plus court chemin entre deux sommets avec l'algorithme de Bellman-Ford
        :param source_vertex: sommet de départ
        :param vertex_order: ordre des sommets à parcourir
        :return: liste des plus court chemins, matrice des distances, nombre d'itérations
    """
    if vertex_order is None:
        vertex_order = self.list_vertex

    # Initialiser la distance du sommet source à 0
    self.distances = np.full(len(self.list_vertex), np.inf)
    self.distances[source_vertex] = 0
    self.paths = [[] for _ in range(len(self.list_vertex))]
    self.paths[0] = [0]

    # Relaxer les arcs répétitivement
    for i in range(len(self.list_vertex)):
        no_updates = True
        for vertex in vertex_order:
            for neighbor, weight in self.graph[vertex]:
                if self.distances[vertex] != np.inf and self.distances[vertex] + weight < self.distances[neighbor]:
                    self.distances[neighbor] = self.distances[vertex] + weight
                    self.paths[neighbor] = self.paths[vertex] + [neighbor]
                    no_updates = False

        self.nb_iter += 1
        if no_updates:
            break

    # Vérifier si il y a un cycle de poids négatif
    contains_negatif_cyle = i >= len(self.list_vertex)-1

    return self.paths, self.distances, self.nb_iter, contains_negatif_cyle