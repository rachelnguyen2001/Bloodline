from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from junction import JunctionTree

def compute_elimination_order(bnet):
    """Computes a low-width elimination order for a Bayesian network.

    YOU DO NOT NEED TO UNDERSTAND HOW THIS FUNCTION WORKS.

    Parameters
    ----------
    bnet : BayesianNetwork
        the Bayesian network for which to compute the elimination order

    Returns
    -------
    list[str]
        the elimination order (a list of the variables of the Bayesian network)
    """

    def build_moral_graph(bnet):
        node_labels = [var for var in bnet.get_variables()]
        edges = []
        for factor in bnet.get_factors():
            vars = [v for v in factor.get_variables()]
            for i, var in enumerate(vars):
                edges += [(var, neighbor) for neighbor in vars[:i] + vars[i + 1:]]
        return UndirectedGraph(len(node_labels), edges, node_labels)

    def min_degree_elim_order(moral_graph):
        def min_degree_node(adjacency):
            best, best_degree = None, float("inf")
            for node in adjacency:
                if len(adjacency[node]) < best_degree:
                    best, best_degree = node, len(adjacency[node])
            return best

        adjacencies = moral_graph.get_adjacencies()
        elim_order = []
        while len(adjacencies) > 0:
            min_degree = min_degree_node(adjacencies)
            adjacencies = {n: adjacencies[n] - {min_degree} for n in adjacencies if n != min_degree}
            elim_order.append(min_degree)
        return elim_order
    moral_graph = build_moral_graph(bnet)
    return min_degree_elim_order(moral_graph), moral_graph


def build_junction_tree(bnet):
    """Constructs a junction tree from a Bayesian network.

    YOU DO NOT NEED TO UNDERSTAND HOW THIS FUNCTION WORKS.

    Parameters
    ----------
    bnet : BayesianNetwork
        the Bayesian network

    Returns
    -------
    JunctionTree
        a reasonably efficient junction tree for the provided Bayesian network
    """

    def elimination_cliques():
        result = []
        adjacencies = moral_graph.get_adjacencies()
        for node in elim_order:
            result.append(adjacencies[node] | {node})
            neighbors = adjacencies[node]
            new_adjacencies = dict()
            for n in adjacencies:
                if n in neighbors:
                    new_adjacencies[n] = (adjacencies[n] - {node}) | (neighbors - {n})
                elif n != node:
                    new_adjacencies[n] = adjacencies[n]
            adjacencies = new_adjacencies
        return result

    elim_order, moral_graph = compute_elimination_order(bnet)
    cliques = elimination_cliques()
    adjacency_matrix = np.zeros((len(cliques), len(cliques)), int)
    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            adjacency_matrix[i][j] = -len(cliques[i] & cliques[j])
    mst = minimum_spanning_tree(adjacency_matrix)
    edges = zip(mst.nonzero()[0], mst.nonzero()[1])
    graph = UndirectedGraph(num_nodes=len(elim_order), node_labels=list(range(len(elim_order))), edges=edges)
    builder = JunctionTreeBuilder(graph, cliques)
    for factor in bnet.get_factors():
        builder.add_factor(factor)
    junction_tree = builder.get_junction_tree()
    return prune_unlabeled_leaves(junction_tree)


class UndirectedGraph:
    """A undirected graph."""

    def __init__(self, num_nodes, edges, node_labels=None):
        self.num_nodes = num_nodes
        if node_labels is None:
            node_labels = [None for _ in range(num_nodes)]
        self.node_labels = node_labels
        self.adjacency = defaultdict(set)
        for (node1, node2) in edges:
            self.adjacency[node1].add(node2)
            self.adjacency[node2].add(node1)
        self.adjacency = dict(self.adjacency)

    def get_neighbors(self, node):
        return list(self.adjacency[node])

    def is_leaf(self, node):
        return len(self.get_neighbors(node)) == 1

    def are_adjacent(self, node1, node2):
        return node2 in self.adjacency[node1]

    def get_adjacencies(self):
        return self.adjacency

    def get_num_nodes(self):
        return self.num_nodes

    def get_node_label(self, index):
        return self.node_labels[index]

    def prune_leaf(self, index):
        assert self.is_leaf(index)
        new_edges = []
        for (x, y) in self.get_edges():
            if x != index and y != index:
                new_edge = [x, y]
                if x > index:
                    new_edge[0] -= 1
                if y > index:
                    new_edge[1] -= 1
                new_edges.append(tuple(new_edge))
        return UndirectedGraph(self.num_nodes-1, new_edges, self.node_labels[:index] + self.node_labels[index+1:])

    def get_edges(self):
        result = set()
        for node in self.adjacency:
            for neighbor in self.adjacency[node]:
                edge = tuple(sorted([node, neighbor]))
                result.add(edge)
        return sorted(result)

    def sprout_leaf(self, node, node_label=None):
        new_node = self.num_nodes
        new_edge = (new_node, node)
        return new_node, UndirectedGraph(self.num_nodes+1,
                                         self.get_edges() + [new_edge],
                                         self.node_labels + [node_label])


    def __str__(self):
        return str(self.get_edges())


def prune_unlabeled_leaves(jtree):
    def first_unlabeled_leaf():
        for node in range(result._graph.get_num_nodes()):
            if result._graph.is_leaf(node) and len(result._factors[node]) == 0:
                return node
        else:
            return None
    result = jtree
    prunable_leaf = first_unlabeled_leaf()
    while prunable_leaf is not None:
        new_graph = result._graph.prune_leaf(prunable_leaf)
        new_factors = result._factors[:prunable_leaf] + result._factors[prunable_leaf+1:]
        result = JunctionTree(new_graph, new_factors)
        prunable_leaf = first_unlabeled_leaf()
    return result



class JunctionTreeBuilder:

    def __init__(self, graph, clusters):
        self.graph = graph
        self.clusters = clusters
        self.node_map = defaultdict(set)
        for node, cluster in enumerate(self.clusters):
            for variable in cluster:
                self.node_map[variable].add(node)
        self.node_map = dict(self.node_map)
        self.factors = defaultdict(list)

    def add_factor(self, factor):
        nodesets = [self.node_map[var] for var in factor.get_variables()]
        possible_assignments = nodesets[0]
        for nodeset in nodesets[1:]:
            possible_assignments = possible_assignments & nodeset
        assignment = list(possible_assignments)[0]
        if not self.graph.is_leaf(assignment):
            assignment, self.graph = self.graph.sprout_leaf(assignment)
        elif len(self.factors[assignment]) > 0:
            assignment1, self.graph = self.graph.sprout_leaf(assignment)
            self.factors[assignment1] = self.factors[assignment]
            self.factors[assignment] = []
            assignment, self.graph = self.graph.sprout_leaf(assignment)
        self.factors[assignment].append(factor)

    def get_junction_tree(self):
        factor_list = [[] for _ in range(self.graph.get_num_nodes())]
        for i, factor in self.factors.items():
            factor_list[i] = factor
        return JunctionTree(self.graph, factor_list)



