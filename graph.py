import numpy as np 

class Graph:
    def __init__(self, liste_sommets):
        self.liste_sommets = liste_sommets
        self.graph = {x : [] for x in self.liste_sommets}
        self.computed_vertices = {}
        self.distances = np.zeros((len(self.liste_sommets)-1,len(self.liste_sommets)))
        self.distances.fill(np.inf)
        self.distances[0][0] = 0 

    def add_edge_weight(self, u, v,w):
        self.graph[u].append((v, w))


    def get_precedent(self, target_vertex):
        liste_precedent = []
        for vertex in self.graph.keys():
            for arcs in self.graph[vertex]:
                if arcs[0] == target_vertex:
                    liste_precedent.append((vertex,arcs[1]))
        return liste_precedent

    # def compute_bellmanford(self,vertex_start,vertex_end):
    #     if self.computed_vertices.get((vertex_start,vertex_end),False):
    #         return self.computed_vertices[(vertex_start,vertex_end)]
        
    #     if vertex_start == vertex_end:
    #         self.computed_vertices[(vertex_start,vertex_end)] = 0
    #         return 0
        
    #     liste_precedent = self.get_precedent(vertex_end)
    #     min_dist = min([self.compute_bellmanford(vertex_start,precedent[0]) + precedent[1] for precedent in liste_precedent])
    #     self.computed_vertices[(vertex_start,vertex_end)] = min_dist
    #     return min_dist

    def compute_bellmanford(self, vertex_start, vertex_end):
        for i in range(1, len(self.liste_sommets)):
            for j, vertex in enumerate(self.liste_sommets):
                for neighbor, weight in self.graph[vertex]:
                    self.distances[i][j] = min(self.distances[i][j], self.distances[i-1][neighbor] + weight)
        return self.distances



