def get_inverse_graph(self):
    """
        <Ajourter la description>
        :return: <Ajourter la description>
    """
    inverse_graph = Graph(self.list_vertex)
    inverse_list_edges = [(v, u, w) for u, v, w in self.list_edges]
    inverse_graph.add_edges(inverse_list_edges)
    return inverse_graph

def get_puits(self):
    """
        <Ajourter la description>
        :return: <Ajourter la description>
    """
    puits = []
    for vertex in self.graph.keys():
        if len(self.graph[vertex]) <= 0:
            puits.append(vertex)

    return puits

def delete_vertex(self, vertex_to_delete : int):
    """
        <Ajourter la description>
        :param vertex_to_delete: <Ajourter la description>
        :return: <Ajourter la description>
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
        <Ajourter la description>
        :param vertex: <Ajourter la description>
        :return: <Ajourter la description>
    """
    if not(vertex in self.graph.keys()):
        return -math.inf
    sum_enter = sum(precedent[1] for precedent in self.get_precedent(vertex))
    sum_exit = sum(neighbor[1] for neighbor in self.graph[vertex])
    return sum_exit - sum_enter