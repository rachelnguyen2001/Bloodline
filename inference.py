from collections import defaultdict
from util import compute_elimination_order, build_junction_tree
from factor import multiply_factors
from bayes import BayesianNetwork

# Change this flag to True once you've implemented belief propagation.
ACTIVATE_BELIEF_PROPAGATION = True

def run_inference(bnet, evidence):
    if ACTIVATE_BELIEF_PROPAGATION:
        return belief_propagation(bnet, evidence)
    else:
        cond_dist = bnet.compute_conditional(["G_elizabeth_ii"], evidence)
        return {'G_elizabeth_ii': cond_dist}

def message_passing(jtree, compute_leaf_msg, compute_msg):
    """Runs a general message-passing algorithm over a junction tree.

    Parameters
    ----------
    jtree : JunctionTree
        the junction tree
    compute_leaf_msg : lambda leaf: ...
        a one-argument function that computes the message that a leaf node (represented
        as its integer index in the junction tree) sends to its only neighbor
    compute_msg : lambda src, dest, msgs: ...
        a three-argument function that computes the message that node src (represented
        as its integer index in the junction tree) sends to node dest; the third
        argument msgs is a set containing the messages sent from src's other neighbors
        (not including dest) to src.

    Returns
    -------
    dict[(int, int), object]
        a dictionary mapping each edge to its message
    """

    def update_ready_list(node):
        if len(waiting_for[node]) == 1:
            neighbor = list(waiting_for[node])[0]
            if (node, neighbor) not in messages:
                ready_to_process.add((node, neighbor))
        elif len(waiting_for[node]) == 0:
            for neighbor in jtree.get_neighbors(node):
                if (node, neighbor) not in messages:
                    ready_to_process.add((node, neighbor))
    messages = dict()
    leaves = [node for node in range(jtree.get_num_nodes()) if jtree.is_leaf(node)]
    for leaf in leaves:
        only_neighbor = jtree.get_neighbors(leaf)[0]
        messages[(leaf, only_neighbor)] = compute_leaf_msg(leaf)
    waiting_for = [set() for _ in range(jtree.get_num_nodes())]
    for (node1, node2) in jtree.get_edges():
        if (node2, node1) not in messages:
            waiting_for[node1].add(node2)
        if (node1, node2) not in messages:
            waiting_for[node2].add(node1)
    ready_to_process = set()
    for node in range(len(waiting_for)):
        update_ready_list(node)
    while len(ready_to_process) > 0:
        src, dest = ready_to_process.pop()
        other_neighbors = set(jtree.get_neighbors(src)) - {dest}
        msgs = [messages[(neighbor, src)] for neighbor in other_neighbors]
        messages[(src, dest)] = compute_msg(src, dest, msgs)
        waiting_for[dest].remove(src)
        update_ready_list(dest)
    return messages

def count_nodes(jtree):
    """Computes the number of nodes in a junction tree.

    This is just an example showing how the general message passing algorithm
    can be used.

    Parameters
    ----------
    jtree : JunctionTree
        the junction tree

    Returns
    -------
    int
        the number of nodes in the junction tree
    """

    messages = message_passing(jtree,
                               lambda leaf: 1,
                               lambda src, dest, msgs: 1 + sum(msgs))
    for (src, dest) in messages:
        if jtree.is_leaf(dest):
            return 1 + messages[(src, dest)]
        
# A recursive helper method that returns a list of variables that appear in factors 
# from all decessors of the src node    
def get_variables_helper(src, dest, jtree):
    result = []
    #if at leaf, return variables in the factor
    if jtree.is_leaf(src):
        return jtree.get_factor(src)._variables
    #else, get the union of the variables of its decessors
    for current in jtree.get_neighbors(src):            
        if current != dest:
            variables = get_variables_helper(current, src, jtree)
            result = list(set().union(result, variables))

    return result
    
#A helper method that returns to the separators    
def get_separator_helper(src, dest, jtree):
    #get the union of variables from each side
    oneside = get_variables_helper(src, dest, jtree)
    otherside = get_variables_helper(dest, src, jtree)
    #get the intersection of the two unions
    to_return = (set(oneside) & set(otherside))
    return to_return

def compute_separators(jtree):
    """Computes the separator of each edge of a junction tree.

    The separator is the set of variables that appear in factors on both sides
    of the edge. In other words, if we removed the edge from the tree, creating
    two separate trees, separator variables are the ones that appear in the
    factors of both trees.

    Parameters
    ----------
    jtree : JunctionTree
        the junction tree

    Returns
    -------
    dict[(int, int), set[int]]
        a dictionary that ssociates each junction tree edge with its separators
    """
    messages = message_passing(jtree, 
                                lambda leaf: get_separator_helper(leaf, jtree.get_neighbors(leaf)[0], jtree),
                                lambda src, dest, msgs: get_separator_helper(src, dest, jtree))
    return messages

def marginalize_helper_for_leaf(leaf, jtree):
    factor = jtree.get_factor(leaf)
    neighbors = jtree.get_neighbors(leaf)
    edge = (leaf, neighbors[0])
    separators = jtree._separators[edge]
    variables = factor._variables

    for var in variables:
        if var not in separators:
            factor = factor.marginalize(var)
    
    return factor

def marginalize_helper_for_non_leaf(src, dest, jtree, msgs, domains):
    separators = jtree._separators[(src, dest)]
    factor = multiply_factors(msgs, domains)
    variables = factor._variables

    for var in variables:
        if var not in separators:
            factor = factor.marginalize(var)
    
    return factor
    
def belief_propagation(bnet, evidence):
    """Computes all single variable distributions, conditioned on the evidence.

    This should return a dictionary that maps each variable v to P(v | evidence).
    These distributions should be computed using the junction tree algorithm
    discussed in class.

    Parameters
    ----------
    bnet : BayesianNetwork
        the Bayesian network
    evidence : dict[str, str]
        the evidence event (represented as a dictionary mapping variables to values)

    Returns
    -------
    dict[str, Factor]
        a dictionary that maps each variable v to P(v | evidence)
    """
    jtree = build_junction_tree(bnet)
    jtree._separators = compute_separators(jtree)
    messages = message_passing(jtree, 
                                lambda leaf: marginalize_helper_for_leaf(leaf, jtree), 
                                lambda src, dest, msgs: marginalize_helper_for_non_leaf(src, dest, jtree, msgs, bnet._domains))
    to_return = {}
    variables = bnet._variables
    for var in variables:
        for (src, dest) in messages:
            if jtree.is_leaf(dest):
                factor = jtree.get_factor(dest)

                #choose a leaf containing the variable
                if var in factor._variables:
                    received_mess = messages[(src, dest)]
                    
                    #multiply its factor with its recieved message
                    new_factor = multiply_factors([factor, received_mess], bnet._domains)
                    new_variables = new_factor._variables

                    #marginalize out other variables
                    for v in new_variables:
                        if v != var:
                            new_factor = new_factor.marginalize(v)

                    to_return[var] = bnet.compute_conditional([var], evidence)
                    
                    
    return to_return