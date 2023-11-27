import numpy as np 

class Graph:
    def __init__(self, liste_sommets):
        self.liste_sommets = liste_sommets
        self.graph = {x-1 : [] for x in self.liste_sommets}
        self.computed_vertices = {}
        self.distances = np.zeros((len(self.liste_sommets)-1,len(self.liste_sommets)))
        self.distances.fill(np.inf)

    def add_edge_weight(self, u, v,w):
        self.graph[u-1].append((v-1, w))


    def get_precedent(self, target_vertex):
        liste_precedent = []
        for vertex in self.graph.keys():
            for arcs in self.graph[vertex]:
                if arcs[0] == target_vertex:
                    liste_precedent.append((vertex,arcs[1]))
        return liste_precedent

    def compute_bellmanford(self, vertex_start, vertex_end):
        self.distances[:,vertex_start] = 0

        for i in range(1, len(self.liste_sommets)-1):
            for vertex in self.liste_sommets:
                vertex -= - 1
                if self.distances[i-1,vertex] == np.inf:
                    continue
                for neighbor, weight in self.graph[vertex]:
                    self.distances[i,neighbor] = min(self.distances[i,neighbor], self.distances[i-1,vertex] + weight)

        return self.distances




