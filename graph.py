import random

import numpy as np
from copy import deepcopy

class Graph:
    def __init__(self, list_vertex):
        # liste des sommets
        self.list_vertex = list_vertex
        # un ordre des sommets 
        self.vertex_order = None
        # représentation du graph sous forme de listes d'adjacence
        self.graph = {x : [] for x in self.list_vertex}
        # distances pour l'algorithm de Bellman-Ford
        self.distances = None
        # prédecesseurs pour l'algorithm de Bellman-Ford
        self.paths = None
        # nombre d'itérations pour l'algorithm de Bellman-Ford
        self.nb_iter = 0
        
    # ajoute une liste d'arcs au graph
    def add_edges(self, liste_arcs):
        for u,v,w in liste_arcs:
            self.add_edge_weight(u,v,w)

    # ajoute un arc entre u et v avec un poids w
    def add_edge_weight(self, u, v, w):
        self.graph[u].append((v, w))

    @staticmethod
    def generate_random_weights(graph):
        """
        :param graphe: Un graphe orienté sans poids
        :return: Creation d'un nouveau graphe avec des poids aléatoirement générés
        """
        random_graph = deepcopy(graph)

        for vertex in random_graph.graph:
            for i, neighboor in enumerate(random_graph.graph[vertex]):
                random_weight = random.randint(-10, 10)
                random_graph.graph[vertex][i] = (neighboor[0], random_weight)

        return random_graph

    # retourne la liste des prédécesseurs de vertex
    def get_precedent(self, target_vertex):
        liste_precedent = []
        for vertex in self.graph.keys():
            for arcs in self.graph[vertex]:
                if arcs[0] == target_vertex:
                    liste_precedent.append((vertex,arcs[1]))
        return liste_precedent

    # calcule le plus court chemin entre vertex_start et vertex_end avec l'algorithme de Bellman-Ford
    def search_bellman_ford(self, source_vertex, vertex_order=None):
        # si vertex_order n'est pas spécifié, on prend l'ordre de la liste des sommets
        if vertex_order is None:
            vertex_order = self.list_vertex

        self.vertex_order = vertex_order
        # initialisation des variables
        self.nb_iter = 0
        nb_vertices = len(vertex_order)
        # initialisation des distances et des prédecesseurs
        self.distances = np.zeros((nb_vertices-1,nb_vertices))
        self.distances.fill(np.inf)
        self.distances[:,source_vertex] = 0
        self.predecessors = np.zeros(self.distances.shape)-1

        # Max itération de l'algorithme de Bellman-Ford
        for i in range(1, len(vertex_order)-1):
            # Pour chaque sommet dans l'ordre donné
            for vertex in vertex_order:
                # Si le sommet n'est pas encore accessible, on passe
                if self.distances[i-1,vertex] == np.inf:
                    continue
                # Sinon, on met à jour les distances pour chaque voisins 
                for neighbor, weight in self.graph[vertex]:
                    new_distance = self.distances[i-1,vertex] + weight
                    if new_distance < self.distances[i,neighbor]:
                        # On met à jour la distance du voisin
                        self.distances[i,neighbor] = new_distance
                        # On met à jour le prédecesseur du voisin
                        self.predecessors[i, neighbor] = vertex

            self.nb_iter += 1
            # # Si les distances n'ont pas changé, on arrête l'algorithme
            if np.array_equal(self.distances[i,:],self.distances[i-1,:]):
                break

        self.paths = self.reconstruct_path(source_vertex)
        return self.paths,self.distances,self.nb_iter
    
    def reconstruct_path(self, source_vertex):
        paths = []
        # Pour chaque sommet
        for vertex in self.vertex_order:
            # Si le sommet n'est pas accessible, on passe
            if self.distances[self.nb_iter,vertex] == np.inf:
                continue
            # Sinon, on reconstruit le chemin
            path = []
            current_vertex = vertex
            # Tant qu'on est pas arrivé au sommet de départ
            while current_vertex != source_vertex:
                # On ajoute le sommet au chemin
                path.append(current_vertex)
                # On récupère le prédecesseur du sommet
                current_vertex = int(self.predecessors[self.nb_iter,current_vertex])
            path.append(source_vertex)
            paths.append(path[::-1])
        return paths
    

    def show_bellmanford_info(self):
        print("nb_iter:")
        print(self.nb_iter)
        print("path:")
        print(self.paths)
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
