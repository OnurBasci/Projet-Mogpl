import numpy as np 
from copy import deepcopy

class Graph:
    def __init__(self, vertex_order):
        # liste des sommets
        self.vertex_order = vertex_order
        # représentation du graph sous forme de listes d'adjacence
        self.graph = {x : [] for x in self.vertex_order}
        # distances pour l'algorithm de Bellman-Ford
        self.distances = None
        # prédecesseurs pour l'algorithm de Bellman-Ford
        self.path = None
        # nombre d'itérations pour l'algorithm de Bellman-Ford
        self.nb_iter = 0
        
    # ajoute une liste d'arcs au graph
    def add_edges(self, liste_arcs):
        for u,v,w in liste_arcs:
            self.add_edge_weight(u,v,w)

    # ajoute un arc entre u et v avec un poids w
    def add_edge_weight(self, u, v, w):
        self.graph[u].append((v, w))

    def add_edges(self, edges):
        for edge in edges:
            self.graph[edge[0]] = (edge[1], edge[2])

    def generate_random_weights(self):
        pass

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
        self.nb_iter = 0
        nb_vertices = len(self.vertex_order)

        self.distances = np.zeros((nb_vertices-1,nb_vertices))
        self.distances.fill(np.inf)
        self.distances[:,vertex_start] = 0
        self.predecessors = np.zeros(self.distances.shape)-1

        for i in range(1, len(self.vertex_order)-1):
            for vertex in self.vertex_order:
                if self.distances[i-1,vertex] == np.inf:
                    continue
                for neighbor, weight in self.graph[vertex]:
                    new_distance = self.distances[i-1,vertex] + weight
                    if new_distance < self.distances[i,neighbor]:
                        self.distances[i,neighbor] = new_distance
                        self.predecessors[i, neighbor] = vertex

            self.nb_iter += 1
            if self.distances[i,:].all() == self.distances[i-1,:].all():
                break

        self.path = self.reconstruct_path(vertex_start,vertex_end)
        return self.path,self.distances,self.nb_iter
    
    def reconstruct_path(self, vertex_start,vertex_end):
        path = [vertex_end]
        while vertex_end != -1 and vertex_end != vertex_start:
            vertex_end = int(self.predecessors[-1,vertex_end])
            if vertex_end != -1:
                path.append(vertex_end)
        if vertex_end == -1:
            path.append(vertex_start)
        return path[::-1]
    
    def show_bellmanford_info(self):
        print("nb_iter:")
        print(self.nb_iter)
        print("path:")
        print(self.path)
        print("distances:")
        print(self.distances)

    def get_path_distance(self):
        return self.path,self.distances

    def get_sources(self):
        sources = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex]) > 0 and len(self.get_precedent(vertex)) <= 0:
                sources.append(vertex)

        return sources

    def get_puits(self):
        puits = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex]) <= 0 and len(self.get_precedent(vertex)) > 0:
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

    def get_diff_enter_exit(self, vertex):
        if not(vertex in self.graph.keys()):
            return 0
        sum_enter = sum(precedent[1] for precedent in self.get_precedent(vertex))
        sum_exit = sum(neighbor[1] for neighbor in self.graph[vertex])
        return sum_exit - sum_enter

    def GloutonFas(self):
        s1 = []
        s2 = []

        graphe = deepcopy(self)
        while len(graphe.graph.keys()) > 0:
            while len(graphe.get_sources()) > 0:
                u = graphe.get_sources()[0]
                s1.append(u)
                graphe.delete_vertex(u)
            while len(graphe.get_puits()) > 0:
                u = graphe.get_puits()[0]
                s2.insert(0, u)
                graphe.delete_vertex(u)

            u_max = np.argmax(np.array([graphe.get_diff_enter_exit(vertex) for vertex in self.graph.keys()]))
            #à changer
            if len(graphe.graph.keys()) <= 1:
                s1.append(list(graphe.graph.keys())[0])
                graphe.graph = {}
            else:
                graphe.delete_vertex(u_max)
                s1.append(u_max)
        s1.extend(s2)
        return s1
