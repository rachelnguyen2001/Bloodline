class JunctionTree:
    """A junction tree."""

    def __init__(self, graph, factors):
        """
        Parameters
        ----------
        graph : UndirectedGraph
            the tree structure of the junction tree
        factors : dict[int, list[Factor]]
            a dictionary mapping leaf nodes to their factors
        """

        self._graph = graph
        self._factors = factors
        self._separators = None

    def get_num_nodes(self):
        """Returns the number of nodes in the junction tree.

        Returns
        -------
        int
            the number of nodes in the junction tree
        """

        return self._graph.get_num_nodes()

    def get_edges(self):
        """Returns the edges in the junction tree.

        Returns
        -------
        list[(int, int)]
            a list of the edges in the junction tree
        """

        return self._graph.get_edges()

    def get_neighbors(self, node):
        """Returns the neighbors of a particular node in the junction tree.

        Parameters
        ----------
        node : int
            the node of interest

        Returns
        -------
        list[int]
            a list of neighbors, i.e. nodes that have an edge connecting them to the node of interest
        """

        return self._graph.get_neighbors(node)

    def get_factor(self, node):
        """Returns the factor associated with a particular node of the junction tree.

        If the node doesn't have an associated factor, then this method returns None.

        Parameters
        ----------
        node : int
            the node of interest

        Returns
        -------
        Factor
             the factor associated with the specified node, or None if there isn't one
        """

        if node < len(self._factors) and len(self._factors[node]) == 1:
            return self._factors[node][0]
        else:
            return None

    def is_leaf(self, node):
        """Returns whether a node is a leaf node.

        Parameters
        ----------
        node : int
            the node of interest

        Returns
        -------
        bool
             True if the node is a leaf, i.e. has exactly one neighbor
        """

        return self._graph.is_leaf(node)

    def __str__(self):
        result = ""
        for node in range(self._graph.get_num_nodes()):
            if self.is_leaf(node):
                result += f'\nnode {node} {self._factors[node]}'
            else:
                result += f'\nnode {node} (neighbors: {self.get_neighbors(node)})'
        return result


