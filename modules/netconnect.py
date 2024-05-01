"""
This module contains the RandomNet class which generates a random graph and checks its connectivity using different methods.
"""
from collections import deque
from typing import Tuple
import networkx as nx
import numpy as np
import time


class RandomNet():
    def __init__(self, n: int, param: float, model: str) -> None:
        """
        This function initializes the RandomNet class.

        Args:
            n (int): Number of nodes
            param (float): Parameter of the random graph model
            model (str): Random graph model
        """
        #Here we store the number of nodes
        self.n = n
        #Here we store the parameter of the random graph model
        self.param = param
        #Here we store the random graph model
        self.model = model
        #Here we generate the random graph
        self.G = self.__get_random__graph_generator()

    def __get_random__graph_generator(self) -> nx.Graph:
        """
        This function generates a random graph based on the random graph model.

        Returns:
            nx.Graph: NetworkX Graph
        """
        #Here we generate the random graph based on the random graph model
        if self.model == 'erdos-renyi':
            return nx.erdos_renyi_graph(self.n, self.param)
        elif self.model == 'r-regular':
            return nx.random_regular_graph(self.param, self.n)
        else:
            raise ValueError('Invalid type')
        

    def check_connectivity(self, method: str) -> Tuple[bool, float]:
        """
        This function checks if a given graph is connected using a given method.

        Args:
            G (nx.Graph): NetworkX Graph
            method (str): Method to check the connectivity

        Returns:
            tuple: Tuple containing the check and the performance of the method
        """
        #Here we store the start time
        start = time.perf_counter()
        
        #Here we check the connectivity of the graph using the given method
        if method == 'bfs':
            check = self.__bfs_connectivity()
        elif method == 'irreducibility':
            check =  self.__irreducibility_connectivity()
        elif method == 'laplacian':
            check =  self.__laplacian_connectivity()
        else:
            raise ValueError('Invalid method')
        
        #Here we store the end time
        end = time.perf_counter()
        #Here we compute the performance of the method
        perf = end-start

        #Here we return the check and the performance
        return check, perf

        
    def __bfs_connectivity(self, source: int = 0) -> bool:
        """
        This function returns the previous and distance dictionaries of a given graph using the Breadth First Search algorithm.

        Args:
            G (nx.Graph): NetworkX Graph
            source (int): Source node id

        Returns:
            bool: True if the graph is connected, False otherwise
        """

        #Here we initialize the distance dictionary which will contain the distance from the source node to each node in the graph
        distances = {node: np.inf for node in self.G.nodes()}

        #Here we initialize the distance of the source node to 0 since the distance from the source node to itself is 0
        distances[source] = 0

        #Here we initialize the queue which will contain the nodes to visit
        queue = deque()
        #Here we append the source node to the queue
        queue.append(source)

        #Here we implement the Breadth First Search algorithm
        #While the queue is not empty, we perform the following steps:
        while len(queue) > 0:
            #We pop the first node of the queue
            node = queue.popleft()
            #Now we iterate over the neighbors of the node
            for neighbor in self.G.neighbors(node):
                #If the distance from the source node to the neighbor is infinite, it means that the neighbor has not been visited yet
                #In this case, we update the distance and the previous node
                if distances[neighbor] == np.inf:
                    #We update the distance
                    distances[neighbor] = distances[node] + 1
                    #Finally, we append the neighbor to the queue
                    queue.append(neighbor)

        #Now we check if the graph is connected. If the distance of any node is infinite, it means that the graph is not connected
        if np.inf in distances.values():
            return False
        else:
            return True
        
    def __irreducibility_connectivity(self) -> bool:
        """
        This function checks if a given graph is connected using the irreducibility of the adjacency matrix.

        Returns:
            bool: True if the graph is connected, False otherwise
        """
        #Here we get the adjacency matrix of the graph with a type of numpy array in order to avoid overflow
        A = nx.to_numpy_array(self.G, dtype=np.int64) 

        #Here we compute the sum I + A + A^2 + ... + A^(n - 1)
        #First we initialize the sum matrix by summing the identity matrix and the adjacency matrix
        sum_matrix = np.identity(self.n, dtype=np.int64) + A
        #We also initialize the A_pow matrix with the adjacency matrix
        A_pow = A
        #Now we iterate over the range from 2 to n and we compute the powers of the matrix A
        for _ in range(2, self.n):
            A_pow = np.matmul(A_pow, A)
            sum_matrix += A_pow

        #If the sum matrix is not greater than 0, it means that the graph is not connected
        if np.all(sum_matrix > 0):
            return True
        else:
            return False
        
    def __laplacian_connectivity(self) -> bool:
        """
        This function checks if a given graph is connected obtaining the second smallest eigenvalue of the Laplacian matrix.

        Returns:
            bool: True if the graph is connected, False otherwise
        """
            
        #Here we get the Laplacian matrix of the graph
        L = nx.laplacian_matrix(self.G).toarray()

        #Here we compute the eigenvalues of the Laplacian matrix
        eigenvalues = np.linalg.eigvals(L)

        #Here we sort the eigenvalues
        sorted_eigenvalues = np.sort(eigenvalues)

        #If the second smallest eigenvalue is greater than 0, it means that the graph is connected
        if sorted_eigenvalues[1] > 0:
            return True
        else:
            return False


            
