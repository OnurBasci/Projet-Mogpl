import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

class Graph:
    """
        Class qui représente un graph orienté avec des arcs pondérés

        :method_static: 
            - show_graph(graph : 'Graph') -> None
            - unifiy_paths(path : list,path2 : list,path3 : list,list_vertex : list) -> 'Graph'
            - generate_random_weights(graph : 'Graph') -> 'Graph'

        :method: 
            - add_edges(liste_edges : list) -> None
            - get_precedent(target_vertex : int) -> list
            - bellman_ford(source_vertex : int , vertex_order : list = None) -> (list,np.ndarray,int)
            - show_bellmanford_result() -> None
            - get_sources() -> list
            - get_puits() -> list
            - delete_vertex(vertex_to_delete : int) -> None
            - get_diff_enter_exit(vertex : int) -> int
            - glouton_fas() -> list
    """
        
    def __init__(self, list_vertex : list):
        # liste des sommets
        self.list_vertex : list = list_vertex
        # liste des arcs
        self.list_edges : list = []
        # un ordre des sommets
        self.vertex_order : list = None
        # représentation du graph sous forme de listes d'adjacence
        self.graph : dict = {x: [] for x in self.list_vertex}
        # distances pour l'algorithm de Bellman-Ford
        self.distances : np.ndarray = np.full(len(self.list_vertex), np.inf)
        # prédecesseurs pour l'algorithm de Bellman-Ford
        self.paths : list = None  
        # nombre d'itérations pour l'algorithm de Bellman-Ford
        self.nb_iter : int = 0

    """
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Fonction static de la classe                                             │  
    └──────────────────────────────────────────────────────────────────────────┘
    """
    @staticmethod
    def show_graph(graph : 'Graph'):
        """
            Fonction qui affiche un graph avec les poids des arcs et les sommets
            :param graph: graph à afficher
            :return: None
        """
        G = nx.DiGraph()
        # Ajout des sommets
        G.add_nodes_from(graph.list_vertex)
        # Ajout des arcs avec les poids
        new_list_edges = [(u,v) for u,v,w in graph.list_edges]
        G.add_edges_from(new_list_edges)
        # Positionnement des sommets
        pos = nx.spring_layout(G)
        # Dessiner les sommets
        nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_color='black', font_size=8)
        # Dessiner les arcs
        edge_labels = {(u, v): f'{weight}' for u, v, weight in graph.list_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.show()


    @staticmethod
    def unifiy_paths(paths : list,list_vertex : list) -> 'Graph':
        """
        Fonction qui unifie les chemins pour créer un graph
        :param paths: chemins des graphes
        :param list_vertex: liste des sommets
        :return: un graph
        """
        # Initialisation des variables
        list_edges : list = []
        added_edges : dict = {}
        # Fonction qui ajoute un arc au graph si il n'existe pas déjà
        def add_edge_from_path(path):
            if len(path) <= 1:
                return
            for i in range(0,len(path) - 1):
                vertex_1,vertex_2 = path[i],path[i+1]
                if added_edges.get((vertex_1,vertex_2),None) is not None:
                    continue
                list_edges.append((path[i],path[i+1],1))
                added_edges[(vertex_1,vertex_2)] = True

        # Ajout des arcs pour chaque chemin
        for p_x in zip(*paths):
            for p in p_x :
                add_edge_from_path(p)
        # Création du graph
        new_graph = Graph(list_vertex)
        new_graph.add_edges(list_edges)
        # Retourne le graph
        return new_graph
    
    @staticmethod
    def generate_graphs_with_random_weights(base_graph : 'Graph', nb_graph:int):
        """
        Fonction qui génère des graphes avec des poids aléatoires
        :param base_graph: Un graphe orienté sans poids
        :param nb_graph: nombre de graph à générer
        :return: liste des graphes générés
        """
        graphs = []
        for i in range(nb_graph):
            graphs.append(Graph.generate_random_weights(base_graph))
        return graphs

    @staticmethod
    def generate_random_weights(graph : 'Graph'):
        """
        Fonction qui génère des poids aléatoires pour un graphe donnée
        :param graph: Un graphe orienté sans poids
        :return: Creation d'un nouveau graphe avec des poids aléatoirement générés
        """

        random_graph = Graph(graph.list_vertex)
        list_edges = []
        for vertex in graph.list_vertex:
            for neighbor, _ in graph.graph[vertex]:
                random_weight = random.randint(-10, 10)
                list_edges.append((vertex, neighbor, random_weight))
        random_graph.add_edges(list_edges)

        paths, _, _, contains_negatif_cycle = random_graph.bellman_ford(0, None)
        path = paths[np.argmax(np.array([len(path) for path in paths]))]
        while contains_negatif_cycle :
            negative_cycle = Graph._find_negative_cycle(path)
            #print(f"{negative_cycle}")
            #print()
            for i in range(len(negative_cycle)-1):
                for u,v,w in list_edges:
                    if u == negative_cycle[i] and v == negative_cycle[i+1]:
                        list_edges.remove((u,v,w))
                        list_edges.append((u,v,abs(w)))
            #last edge between last and first vertexes
            for u, v, w in list_edges:
                if u == negative_cycle[-1] and v == negative_cycle[0]:
                    list_edges.remove((u, v, w))
                    list_edges.append((u, v, abs(w)))
            random_graph.add_edges(list_edges)

            paths, _, _, contains_negatif_cycle = random_graph.bellman_ford(0, None)
            path = paths[np.argmax(np.array([len(path) for path in paths]))]
        return random_graph

    
    @staticmethod
    def generate_random_order(graph):
        """
        Fonction qui génère un ordre aléatoire pour un graphe donnée
        """
        nb_elem = len(graph.list_vertex)
        return random.sample(range(nb_elem), nb_elem)
    
    @staticmethod
    def generate_random_graph(size_graph : int,nb_edges : int) -> 'Graph':
        """
            Fonction qui génère un graph aléatoire d'une taile donnée
        """
        liste_vertex : list = [i for i in range(size_graph)]
        liste_edges = [tuple(random.sample(liste_vertex,2)+[1]) for _ in range(nb_edges)]
        new_graph = Graph(liste_vertex)
        new_graph.add_edges(liste_edges)
        new_graph = Graph.generate_random_weights(new_graph)
        return new_graph

    @staticmethod
    def generate_level_graph(nb_level):
        """
        This function creates a graph by levels as explained in the guestion 11
        """
        level_graph = Graph([i for i in range(nb_level * 4)])

        edges = []
        for i in range(nb_level-1):
            next_level_vertexes = [(i*4)+4+k for k in range(4)]
            for j in range(4):
                for k in next_level_vertexes:
                    edges.append(((i*4)+j, k, 1))
        level_graph.add_edges(edges)
        level_graph = Graph.generate_random_weights(level_graph)

        return level_graph

    
    @staticmethod
    def generate_compare_graph(size_graph:int,nb_edges:int,nb_graph_to_generate:int):
        """
            Fonction qui compare les résultats de l'ordre glouton_fas avec un ordre aléatoire
            :param size_graph: taille du graph
            :param nb_edges: nombre d'arcs
            :param nb_graph_to_generate: nombre de graph à générer
        """
        source = 0
        # Génération d'un graph aléatoire
        graph = Graph.generate_random_graph(size_graph=size_graph,nb_edges=nb_edges)
        # Génération de graphes avec des poids aléatoires
        train_test_graphs = Graph.generate_graphs_with_random_weights(graph,nb_graph_to_generate)
        # Récupération des graphes d'entrainement et de test
        train_graphs = train_test_graphs[:nb_graph_to_generate-1]
        test_graph = train_test_graphs[nb_graph_to_generate-1]
        # Calcul de l'arborecence 
        paths = [graph.bellman_ford(source,None)[0] for graph in train_graphs]
        # Unifier les arborecences
        union_graph = Graph.unifiy_paths(paths,graph.list_vertex)
        # Calcul de l'ordre avec glouton_fas
        order = union_graph.glouton_fas()
        # Génération d'un ordre aléatoire
        random_order = Graph.generate_random_order(graph)
        # Calcul de l'arborecence avec l'ordre glouton_fas
        _,_,nb_iter_glouton,_ = test_graph.bellman_ford(source,order)
        _,_,nb_iter_random,_ = test_graph.bellman_ford(source,random_order)


        print(f"Ordre glouton_fas: {order}")
        print(f"Nombre d'itérations avec glouton_fas: {nb_iter_glouton}")
        print(f"Ordre aléatoire: {random_order}")
        print(f"Nombre d'itérations avec un ordre aléatoire: {nb_iter_random}")

        return nb_iter_glouton, nb_iter_random

    @staticmethod
    def compare_graph(graph, nb_graph_to_generate):
        source = 0
        # Génération de graphes avec des poids aléatoires
        train_test_graphs = Graph.generate_graphs_with_random_weights(graph, nb_graph_to_generate)
        train_graphs = train_test_graphs[:nb_graph_to_generate - 1]
        test_graph = train_test_graphs[nb_graph_to_generate - 1]
        # Calcul de l'arborecence
        paths = [graph.bellman_ford(source, None)[0] for graph in train_graphs]
        # Unifier les arborecences
        union_graph = Graph.unifiy_paths(paths, graph.list_vertex)
        # Calcul de l'ordre avec glouton_fas
        order = union_graph.glouton_fas()
        # Génération d'un ordre aléatoire
        random_order = Graph.generate_random_order(graph)
        # Calcul de l'arborecence avec l'ordre glouton_fas
        _, _, nb_iter_glouton, _ = test_graph.bellman_ford(source, order)
        _, _, nb_iter_random, _ = test_graph.bellman_ford(source, random_order)

        """
        print(f"Ordre glouton_fas: {order}")
        print(f"Nombre d'itérations avec glouton_fas: {nb_iter_glouton}")
        print(f"Ordre aléatoire: {random_order}")
        print(f"Nombre d'itérations avec un ordre aléatoire: {nb_iter_random}")"""

        return nb_iter_glouton, nb_iter_random

    @staticmethod
    def _find_negative_cycle( path : list) -> int:
        """
            Fonction qui trouve un cycle à poids négatif
            :param path: chemin
            :return: nombre d'itérations
        """
        already_seen = {}
        index_cycle = -1
        for i,value in enumerate(path):
            if value in already_seen:
                index_cycle = i
                break
            already_seen[value] = i
        i_start_cycle = already_seen[path[index_cycle]]
        cycle = path[i_start_cycle:index_cycle]
        return cycle

    """
    ┌──────────────────────────────────────────────────────────────────────────┐
    │ Fonction de la classe                                                    │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    def add_edges(self, liste_edges : list) -> None:
        """
            Fonction qui ajoute des arcs au graph
            :param liste_edges: liste des arcs
            :return: None
        """
        self.list_edges = liste_edges
        self.graph = {x: [] for x in self.list_vertex}
        for u,v,w in liste_edges:
            self.graph[u].append((v, w))

    def get_precedent(self, target_vertex : int) -> list:
        """
            Fonction qui retourne la liste des prédecesseurs d'un sommet
            :param target_vertex: sommet cible
            :return: liste des prédecesseurs du sommet cible

        """
        liste_precedent : list = []
        # Pour chaque sommet
        for vertex in self.graph.keys():
            # Pour chaque arc du sommet
            for v,w in self.graph[vertex]:
                # Si l'arc arrive sur le sommet cible, on l'ajoute à la liste des prédecesseurs
                if v == target_vertex:
                    liste_precedent.append((vertex,w))
        return liste_precedent

    def bellman_ford(self, source_vertex, vertex_order=None):
        self.nb_iter = 0
        """
            Fonction qui calcule le plus court chemin entre deux sommets avec l'algorithme de Bellman-Ford
            :param source_vertex: sommet de départ
            :param vertex_order: ordre des sommets à parcourir
            :return: liste des plus court chemins, matrice des distances, nombre d'itérations
        """
        if vertex_order is None:
            vertex_order = self.list_vertex

        # Initialiser la distance du sommet source à 0
        self.distances = np.full(len(self.list_vertex), np.inf)
        self.distances[source_vertex] = 0
        self.paths = [[] for _ in range(len(self.list_vertex))]
        self.paths[0] = [0]
        predecessor = [-1] * len(self.list_vertex)

        # Relaxer les arcs répétitivement
        for i in range(len(self.list_vertex)):
            no_updates = True
            for vertex in vertex_order:
                for neighbor, weight in self.graph[vertex]:
                    if self.distances[vertex] != np.inf and self.distances[vertex] + weight < self.distances[neighbor]:
                        self.distances[neighbor] = self.distances[vertex] + weight
                        self.paths[neighbor] = self.paths[vertex] + [neighbor]
                        predecessor[neighbor] = vertex
                        no_updates = False

            self.nb_iter += 1
            if no_updates:
                break
        contains_negatif_cyle = False

        #check if there is a negatif cycle
        if i >= len(self.list_vertex)-1:
            contains_negatif_cyle = True

        return self.paths, self.distances, self.nb_iter, contains_negatif_cyle

    def bellman_fordv2(self, source_vertex, vertex_order=None):
        self.nb_iter = 0
        """
            Fonction qui calcule le plus court chemin entre deux sommets avec l'algorithme de Bellman-Ford
            :param source_vertex: sommet de départ
            :param vertex_order: ordre des sommets à parcourir
            :return: liste des plus court chemins, matrice des distances, nombre d'itérations
        """
        if vertex_order is None:
            vertex_order = self.list_vertex

        # Initialiser la distance du sommet source à 0
        self.distances[source_vertex] = 0
        self.paths =  [[source_vertex] for _ in range(len(self.list_vertex))]
        predecessor = [-1] * len(self.list_vertex)

        # Relaxer les arcs répétitivement
        for _ in range(len(self.list_vertex) - 1):
            no_updates = True
            for vertex in vertex_order :
                for neighbor, weight in self.graph[vertex]:
                    if self.distances[vertex] != np.inf and self.distances[vertex] + weight < self.distances[neighbor]:
                        self.distances[neighbor] = self.distances[vertex] + weight
                        self.paths[neighbor] = self.paths[vertex] + [neighbor]
                        predecessor[neighbor] = vertex
                        no_updates = False 

            self.nb_iter += 1
            if no_updates:
                break
            
        # Vérifier les cycles de poids négatifs
        """for vertex, neighbor, weight in self.list_edges:
            if self.distances[vertex] != np.inf and self.distances[vertex] + weight < self.distances[neighbor]:
                return [], [], 0, False"""

        for u, v, w in self.list_edges:
            if self.distances[u] != np.inf and self.distances[u] + w < self.distances[v]:
                # Negative cycle detected
                visited = [False] * len(self.list_vertex)
                visited[v] = True

                while not visited[u]:
                    visited[u] = True
                    u = predecessor[u]

                # u is a vertex in a negative cycle, find the cycle itself
                negative_cycle = [u]
                v = predecessor[u]

                while v != u:
                    negative_cycle = [v] + negative_cycle
                    v = predecessor[v]

                return negative_cycle,[],[],False



        return self.paths, self.distances, self.nb_iter, True

    def show_bellmanford_result(self):
        """
            Fonction qui affiche les résultats de l'algorithme de Bellman-Ford
            :return: None
        """
        print("=========================\n")	
        print(f"[INFO] BellmanFord completed in {self.nb_iter} iterations")
        print(f"[INFO] Paths found : {self.paths}")
        print(f"[INFO] Matrix distance : \n {self.distances}")
        print("\n")

    def get_sources(self):
        """
            <Ajourter la description>
            :return: <Ajourter la description>
        """
        sources = []
        for vertex in self.graph.keys():
            if len(self.get_precedent(vertex)) <= 0:
                sources.append(vertex)

        return sources

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

    def glouton_fas(self):
        """
            <Ajourter la description>
            :return: <Ajourter la description>
        """

        s1 : list = []
        s2 : list = []
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

            if len(graphe.graph.keys()) > 0:
                s1.append(u_max)
                graphe.delete_vertex(u_max)
        s1.extend(s2)
        return s1



    def delete_cycles(self, vertex_order=None):
        """
        remove negatif cycles until there is none
        """
        accepted = False
        while not accepted:
            cycle, _, _, state = self.bellman_ford(0, vertex_order)
            if state:
                accepted = True
            else:
                # print("Cycle négatif détecté, re-génération des poids")
                # print(f"Cycle: {cycle}")
                self.delete_negatif_cyle(cycle)
    def delete_negatif_cyle(self, cycle):
        """
        set all weights of edges between the vertex of the cycle and it's neighboors to positif
        """
        for k, bad_vertex in enumerate(cycle):
            for i, neighboor in enumerate(self.graph[bad_vertex]):
                # print(f"bad weight: {random_graph.graph[bad_vertex][i][1]}")
                v, w = self.graph[bad_vertex][i]
                self.graph[bad_vertex][i] = (v, abs(w))
