def get_inverse_graph(self):
        """
        :return: Un graphe avec les poids inversés
        """
        inverse_graph = Graph(self.list_vertex)
        inverse_list_edges = [(v, u, w) for u, v, w in self.list_edges]
        inverse_graph.add_edges(inverse_list_edges)
        return inverse_graph

def get_puits(self):
    """
        Méthode qui calcule les puits dans l'attribut graphe i.e. les sommet qui n'ont pas de voisins mais des précedents
        :return: Une liste de précedents
    """
    puits = []
    for vertex in self.graph.keys():
        if len(self.graph[vertex]) <= 0:
            puits.append(vertex)

    return puits

def delete_vertex(self, vertex_to_delete : int):
    """
        Methode qui supprime un sommet
        :param vertex_to_delete: sommet à supprimer
        :return: None
    """
    #remove vertex
    del self.graph[vertex_to_delete]
    #remove precedents
    for vertex in self.graph.keys():
        for arcs in self.graph[vertex]:
            if arcs[0] == vertex_to_delete:
                self.graph[vertex].remove(arcs)

def get_diff_enter_exit(self, vertex : int):
    """
        Méthode calculant la difference entre les valeurs entrant et la valeur sortant d'un sommet dans l'attribut graph
        :param vertex: Le sommet à calculer la difference
        :return: La différence
    """
    if not(vertex in self.graph.keys()):
        return -math.inf
    sum_enter = sum(precedent[1] for precedent in self.get_precedent(vertex))
    sum_exit = sum(neighbor[1] for neighbor in self.graph[vertex])
    return sum_exit - sum_enter