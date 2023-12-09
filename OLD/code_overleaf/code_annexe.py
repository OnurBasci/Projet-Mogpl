def get_sources(self):
    """
        Méthode qui calcule les sources dans l'attribut graphe i.e. les sommet qui n'ont pas de précedents mais des voisins
        :return: Une liste de sources
    """
    sources = []
    for vertex in self.graph.keys():
        if len(self.get_precedent(vertex)) <= 0:
            sources.append(vertex)

    return sources