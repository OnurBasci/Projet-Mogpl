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
            - unifiy_paths(paths : list,list_vertex : list) -> 'Graph'
            - generate_graphs_with_random_weights(base_graph : 'Graph', nb_graph:int) -> list['Graph']
            - generate_random_weights(graph : 'Graph') -> 'Graph'
            - generate_random_order(graph : 'Graph') -> list
            - generate_random_graph(size_graph : int,nb_edges : int) -> 'Graph'
            - generate_level_graph(nb_level) -> 'Graph'
            - generate_compare_graph(size_graph:int,nb_edges:int,nb_graph_to_generate:int) -> (int,int)
            - compare_graph(graph, nb_graph_to_generate) -> (int,int)
            - _find_negative_cycle( path : list) -> int


        :method: 
            - add_edges(self, liste_edges : list) -> None
            - get_precedent(self, target_vertex : int) -> list
            - bfs(self,vertex_source:int)->int
            - bellman_ford(self, source_vertex, vertex_order=None) -> (list,np.ndarray,int,bool)
            - show_bellmanford_result(self) -> None
            - get_sources(self) -> list
            - get_puits(self) -> list
            - delete_vertex(self, vertex_to_delete : int) -> None
            - get_diff_enter_exit(self, vertex : int) -> int
            - glouton_fas(self) -> list
            - glouton_fas_v2(self) -> list
            - get_inverse_graph(self) -> 'Graph'
            
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
        # arborescence des plus courts chemins
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
    def generate_graphs_with_random_weights(base_graph : 'Graph', nb_graph:int)-> list['Graph']:
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

        # Génération des poids aléatoires entre -10 et 10
        random_graph = Graph(graph.list_vertex)
        list_edges = []
        for vertex in graph.list_vertex:
            for neighbor, _ in graph.graph[vertex]:
                random_weight = random.randint(-10, 10)
                list_edges.append((vertex, neighbor, random_weight))
        random_graph.add_edges(list_edges)

        # Vérification qu'il n'y a pas de cycle négatif
        paths, _, _, contains_negatif_cycle = random_graph.bellman_ford(0, None)
        path = paths[np.argmax(np.array([len(path) for path in paths]))]
        # Si il y a un cycle négatif, on le supprime
        while contains_negatif_cycle :
            # On récupère le cycle négatif
            negative_cycle = Graph._find_negative_cycle(path)
            # On convertit les poids négatifs en positif
            for i in range(len(negative_cycle)-1):
                for u,v,w in list_edges:
                    if u == negative_cycle[i] and v == negative_cycle[i+1]:
                        list_edges.remove((u,v,w))
                        list_edges.append((u,v,abs(w)))

            # On fait la même chose pour le dernier arc
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
            Fonction qui génère des poids aléatoires pour un graphe donnée
            :param size_graph: Le nombre de sommet
            :return: La fonction générée avec des poids aléatoire
        """
        not_accepted = True
        while not_accepted:
            liste_vertex : list = [i for i in range(size_graph)]
            liste_edges = [tuple(random.sample(liste_vertex,2)+[1]) for _ in range(nb_edges)]
            new_graph = Graph(liste_vertex)
            new_graph.add_edges(liste_edges)
            new_graph = Graph.generate_random_weights(new_graph)
            nb_vertex = new_graph.bfs(0)
            if nb_vertex >= size_graph//2:
                not_accepted = False
        
        return new_graph

    @staticmethod
    def generate_level_graph(nb_level):
        """
            Cette fonction crée un graph avec des niveau comme expliqué dans la question 11
            :param nb_level: nombre de niveau
            :level_graph: La fonction générée de la structure expliqué dans la question 11
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
        order = union_graph.glouton_fas_v2()
        # Génération d'un ordre aléatoire
        random_order = Graph.generate_random_order(graph)
        # Calcul de l'arborecence avec l'ordre glouton_fas
        _,dist,nb_iter_glouton,_ = test_graph.bellman_ford(source,order)
        _,dist_random,nb_iter_random,_ = test_graph.bellman_ford(source,random_order)


        # print(f"Ordre glouton_fas: {order}")
        # print(f"Nombre d'itérations avec glouton_fas: {nb_iter_glouton}")
        # print(f"Ordre aléatoire: {random_order}")
        # print(f"Nombre d'itérations avec un ordre aléatoire: {nb_iter_random}")

        # On verifie qu'on à bien le même résultat avec les deux ordres
        if(dist.any() != dist_random.any()):
            print("[ERREUR]: les distances sont différentes")
            print(f"Distance avec glouton_fas: {dist}")
            print(f"Distance avec un ordre aléatoire: {dist_random}")

        return nb_iter_glouton, nb_iter_random  

    @staticmethod
    def compare_graph(graph, nb_graph_to_generate):
        """
        Cette méthode compare les nombres d'itération nécessaire pour appliquer l'algorithme de Bellman Ford
        suivant l'ordre obtenu avec glouton fas et un ordre aléatoire.
        :param graph: Le graphe à analyser le nombre d'itération.
        :param Nombre de graphe d'entrâinement pour obtenir l'ordre de glouton fas.
        :return: nombre d'itération obtenus avec 2 méthodes
        """
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
        order = union_graph.glouton_fas_v2()
        # Génération d'un ordre aléatoire
        random_order = Graph.generate_random_order(graph)
        # Calcul de l'arborecence avec l'ordre glouton_fas
        _, dist, nb_iter_glouton, _ = test_graph.bellman_ford(source, order)
        _, dist_random, nb_iter_random, _ = test_graph.bellman_ford(source, random_order)

        if(dist.any() != dist_random.any()):
            print("[ERREUR]: les distances sont différentes")
            print(f"Distance avec glouton_fas: {dist}")
            print(f"Distance avec un ordre aléatoire: {dist_random}")


        """print(f"Ordre glouton_fas: {order}")
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

    def bfs(self,vertex_source:int)->int:
        """
            Fonction qui parcours le graphe en largeur
            :param vertex_source: sommet de départ
            :return: nombre de sommets accessibles depuis le sommet source de départ
        """
        # Initialisation des variables
        visited = [False for _ in range(len(self.list_vertex))]
        queue = []
        nb_vertex = 0
        # On ajoute le sommet de départ à la file
        queue.append(vertex_source)
        visited[vertex_source] = True
        # Tant que la file n'est pas vide
        while queue:
            # On récupère le sommet de la file
            vertex = queue.pop(0)
            nb_vertex += 1
            # Pour chaque voisin du sommet
            for neighbor, _ in self.graph[vertex]:
                # Si le voisin n'a pas été visité, on l'ajoute à la file
                if not visited[neighbor]:
                    queue.append(neighbor)
                    visited[neighbor] = True

        return nb_vertex-1
        

    def bellman_ford(self, source_vertex, vertex_order=None):
        """
            Fonction qui calcule le plus court chemin entre deux sommets avec l'algorithme de Bellman-Ford
            :param source_vertex: sommet de départ
            :param vertex_order: ordre des sommets à parcourir
            :return: liste des plus court chemins, matrice des distances, nombre d'itérations
        """
        self.nb_iter = 0
        if vertex_order is None:
            vertex_order = self.list_vertex

        # Initialiser la distance du sommet source à 0
        self.distances = np.full(len(self.list_vertex), np.inf)
        self.distances[source_vertex] = 0
        self.paths = [[] for _ in range(len(self.list_vertex))]
        self.paths[source_vertex] = [source_vertex]

        # Relaxer les arcs répétitivement
        for i in range(len(self.list_vertex)):
            no_updates = True
            for vertex in vertex_order:
                for neighbor, weight in self.graph[vertex]:
                    if self.distances[vertex] != np.inf and self.distances[vertex] + weight < self.distances[neighbor]:
                        self.distances[neighbor] = self.distances[vertex] + weight
                        self.paths[neighbor] = self.paths[vertex] + [neighbor]
                        no_updates = False

            self.nb_iter += 1
            if no_updates:
                break

        # Vérifier si il y a un cycle de poids négatif
        contains_negatif_cyle = i >= len(self.list_vertex)-1

        return self.paths, self.distances, self.nb_iter, contains_negatif_cyle

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
            Méthode qui calcule les sources dans l'attribut graphe i.e. les sommet qui n'ont pas de précedents mais des voisins
            :return: Une liste de sources
        """
        sources = []
        for vertex in self.graph.keys():
            if len(self.get_precedent(vertex)) <= 0:
                sources.append(vertex)

        return sources


    def get_puits(self):
        """
            Méthode qui calcule les puits dans l'attribut graphe i.e. les sommet qui n'ont pas de voisins mais des précedents
            :return: Une liste de précedents
        """
        puits = []
        for vertex in self.graph.keys():
            if len(self.graph[vertex]) <= 0:
                puits.append(vertex)

        return puits

    def delete_vertex(self, vertex_to_delete : int):
        """
            Methode qui supprime un sommet
            :param vertex_to_delete: sommet à supprimer
            :return: None
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
            Méthode calculant la difference entre les valeurs entrant et la valeur sortant d'un sommet dans l'attribut graph
            :param vertex: Le sommet à calculer la difference
            :return: La différence
        """
        if not(vertex in self.graph.keys()):
            return -math.inf
        sum_enter = sum(precedent[1] for precedent in self.get_precedent(vertex))
        sum_exit = sum(neighbor[1] for neighbor in self.graph[vertex])
        return sum_exit - sum_enter

    def glouton_fas(self):
        """
            La méthode glouton qui calcule un ordre total à partir de l'attribut graph de la classe
            :return: (List) ordre
        """

        s1 : list = []
        s2 : list = []
        graphe = deepcopy(self)
        while len(graphe.graph.keys()) > 0:
            sources = graphe.get_sources()
            while len(sources) > 0:
                u = sources[0]
                s1.append(u)
                graphe.delete_vertex(u)
                sources = graphe.get_sources()
            puits = graphe.get_puits()
            while len(puits) > 0:
                puits = graphe.get_puits()
                u = puits[0]
                s2.insert(0, u)
                graphe.delete_vertex(u)
            u_max = np.argmax(np.array([graphe.get_diff_enter_exit(vertex) for vertex in self.graph.keys()]))

            if len(graphe.graph.keys()) > 0:
                s1.append(u_max)
                graphe.delete_vertex(u_max)
        s1.extend(s2)
        return s1

    def glouton_fas_v2(self):
        """
        La méthode glouton qui calcule un ordre total à partir de l'attribut graph de la classe
        :return: Un ordre
        """
        s1: list = []
        s2: list = []
        graphe = deepcopy(self)
        # Inverse le graphe pour faciliter la recherche
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

    def get_inverse_graph(self):
        """
        :return: Un graphe avec les poids inversés
        """
        inverse_graph = Graph(self.list_vertex)
        inverse_list_edges = [(v, u, w) for u, v, w in self.list_edges]
        inverse_graph.add_edges(inverse_list_edges)
        return inverse_graph
