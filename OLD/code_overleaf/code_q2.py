def glouton_fas(self):
    s1: list = []
    s2: list = []
    graphe = deepcopy(self)
    inverse_graphe = graphe.get_inverse_graph()
    while len(graphe.graph.keys()) > 0:
        sources = inverse_graphe.get_puits()
        while len(sources) > 0:
            u = sources[0]
            s1.append(u)
            graphe.delete_vertex(u)
            inverse_graphe.delete_vertex(u)
            sources = inverse_graphe.get_puits()
        puits = graphe.get_puits()
        while len(puits) > 0:
            u = puits[0]
            s2.insert(0, u)
            graphe.delete_vertex(u)
            puits = graphe.get_puits()
        u_max = np.argmax(np.array([graphe.get_diff_enter_exit(vertex) for vertex in self.graph.keys()]))

        if len(graphe.graph.keys()) > 0:
            s1.append(u_max)
            graphe.delete_vertex(u_max)
    s1.extend(s2)
    return s1