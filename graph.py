import numpy as np 
from copy import deepcopy

class Graph:
    def __init__(self, liste_sommets):
        self.liste_sommets = liste_sommets
        self.graph = {x : [] for x in self.liste_sommets}
        self.computed_vertices = {}
        self.distances = np.zeros((len(self.liste_sommets)-1,len(self.liste_sommets)))
        self.distances.fill(np.inf)
        self.distances[:,0] = 0 

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
        for i in range(1, len(self.liste_sommets)-1):
            for vertex in self.liste_sommets:
                for neighbor, weight in self.graph[vertex]:
                    self.distances[i][vertex] = min(self.distances[i][vertex], self.distances[i-1][neighbor] + weight)
        return self.distances

    def get_sources(self):
        sources = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex] > 0 and len(self.get_precedent(vertex)) <= 0):
                sources.append(vertex)

        return sources

    def get_puits(self):
        puits = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex] <= 0 and len(self.get_precedent(vertex)) > 0):
                puits.append(vertex)

        return puits

    def delete_vertex(self, vertex_to_delete):
        #remove vertex
        del self.graph[vertex_to_delete]
        #remove precedents
        for vertex in self.graph.keys():
            for arcs in self.graph[vertex]:
                if arcs[0] == vertex_to_delete:
                    self.graph[vertex].remove(arcs)

    def get_diff_enter_exit(self, vertex, graph):
        sum_enter = sum(precedent[1] for precedent in graph.get_precedent(vertex))
        sum_exit = sum(neighbor[1] for neighbor in graph[vertex])
        return sum_exit - sum_enter

    def GloutonFas(self):
        s1 = []
        s2 = []

        graphe = deepcopy(self.graph)
        while len(graphe.keys()) > 0:
            while len(self.get_sources()) > 0:
                u = self.get_sources()[0]
                s1.insert(u, 0)
                graphe.delete_vertex(u)
            while len(self.get_puits()) > 0:
                u = self.get_puits()[0]
                s2.append(u)
                graphe.delete_vertex(u)

            u_max = np.argmax(np.array([self.get_diff_enter_exit(vertex, graphe) for vertex in graphe.keys()]))
            s1.append(u_max)
            graphe.delete_vertex(u_max)

        return s1.extend(s2)



