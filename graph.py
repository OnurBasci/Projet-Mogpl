import numpy as np 
from copy import deepcopy

class Graph:
    def __init__(self, liste_vertex):
        # liste des sommets
        self.liste_vertex = liste_vertex
        # représentation du graph sous forme de listes d'adjacence
        self.graph = {x : [] for x in self.liste_vertex}
        # distances pour l'algorithm de Bellman-Ford
        self.distances = None
        # prédecesseurs pour l'algorithm de Bellman-Ford
        self.path = None

    # ajoute un arc entre u et v avec un poids w
    def add_edge_weight(self, u, v,w):
        self.graph[u].append((v, w))

    # retourne la liste des prédécesseurs de vertex
    def get_precedent(self, target_vertex):
        liste_precedent = []
        for vertex in self.graph.keys():
            for arcs in self.graph[vertex]:
                if arcs[0] == target_vertex:
                    liste_precedent.append((vertex,arcs[1]))
        return liste_precedent

    # calcule le plus court chemin entre vertex_start et vertex_end avec l'algorithme de Bellman-Ford
    def compute_bellmanford(self, vertex_start, vertex_end):
        nb_iter = 0
        nb_vertices = len(self.liste_vertex)

        self.distances = np.zeros((nb_vertices-1,nb_vertices))
        self.distances.fill(np.inf)
        self.distances[:,vertex_start] = 0
        self.predecessors = np.zeros(self.distances.shape)-1

        for i in range(1, len(self.liste_vertex)-1):
            for vertex in self.liste_vertex:
                if self.distances[i-1,vertex] == np.inf:
                    continue
                for neighbor, weight in self.graph[vertex]:
                    new_distance = self.distances[i-1,vertex] + weight
                    if new_distance < self.distances[i,neighbor]:
                        self.distances[i,neighbor] = new_distance
                        self.predecessors[i, neighbor] = vertex

            if self.distances[i,vertex_end] == np.inf:
                nb_iter += 1 

        self.path = self.reconstruct_path(vertex_start,vertex_end)
        return self.path,self.distances,nb_iter
    
    def reconstruct_path(self, vertex_start,vertex_end):
        path = [vertex_end]
        print(self.predecessors)
        while vertex_end != -1 and vertex_end != vertex_start:
            vertex_end = int(self.predecessors[-1,vertex_end])
            if vertex_end != -1:
                path.append(vertex_end)
        if vertex_end == -1:
            path.append(vertex_start)
        return path[::-1]
    
    def get_path_distance(self):
        return self.path,self.distances


