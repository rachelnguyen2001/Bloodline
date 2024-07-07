from util import UndirectedGraph
from factor import multiply_factors
from util import compute_elimination_order


class BayesianNetwork:
    """Represents a Bayesian network by its factors (i.e. the CPTs).
    Parameters
    ----------
    factors : list[factor.Factor]
        The factors of the Bayesian network
    domains : dict[str, list[str]]
        A dictionary mapping each variable to its possible values
    """

    def __init__(self, factors, domains):
        self._factors = factors
        self._domains = domains
        self._variables = set()
        for factor in self._factors:
            self._variables = self._variables | set(factor.get_variables())

    def get_variables(self):
        """Returns the set of variables that appear in at least one factor."""
        return self._variables

    def get_domains(self):
        """Returns the variable signature associated with the Bayesian network."""
        return self._domains

    def get_factors(self):
        """Returns the factors of the Bayesian network."""
        return self._factors

    def eliminate(self, variable):
        """Eliminates a variable from the Bayesian network.
        By "eliminate", we mean that the factors containing the variable are multiplied,
        and then the variable is marginalized (summed) out of the resulting factor.
        Parameters
        ----------
        variable : str
            the variable to eliminate from the Bayesian network
        Returns
        -------
        BayesianNetwork
            a new BayesianNetwork, equivalent to the current Bayesian network, after
            eliminating the specified variable
        """
        # question four
        # list factors that contain the specific variable we are eliminating in the network
        contained_factors = []
        not_contained_factors = []

        for factor in self._factors:
            factor_variables = factor.get_variables()
            
            if variable in factor_variables:
                contained_factors.append(factor)
            else:
                not_contained_factors.append(factor)
                
        # multiply factors containing the variable we are eliminating
        multiplied_factor = multiply_factors(contained_factors, self._domains)
        # marginalize variable out of the multiplied factor
        marginalized_factor = multiplied_factor.marginalize(variable)
        #rebuild the list of factors in the Bayesian Network
        not_contained_factors.append(marginalized_factor)
        
        #update the domains
        new_domains = {}
        for i in self._domains:
            if i is not variable:
                new_domains[i] = self._domains[i]
        
        return BayesianNetwork(not_contained_factors, new_domains)


    def compute_marginal(self, vars):
        """Computes the marginal probability over the specified variables.
        This method uses variable elimination to compute the marginal distribution.
        Parameters
        ----------
        vars : set[str]
            the variables that we want to compute the marginal over
        """

        elim_order, _ = compute_elimination_order(self)
        bnet = self
        revised_elim_order = [var for var in elim_order if var not in vars]
        for var in revised_elim_order:
            bnet = bnet.eliminate(var)
        return multiply_factors(bnet.get_factors(), bnet.get_domains())

    def compute_conditional(self, vars, evidence):
        """Computes the conditional distibution over a set of variables given an evidence event.
        Parameters
        ----------
        vars : list[str]
            the variables that we want to compute the probability distribution over
        evidence : dict[str, str]
            the observed event
        Returns
        -------
        float
            the conditional probability of the event according to the Bayesian network
        """

        all_vars = list(vars) + list(evidence.keys())
        marginal = self.compute_marginal(all_vars)
        marginal = marginal.reduce(evidence)
        for var in evidence:
            marginal = marginal.marginalize(var)
        return marginal.normalize()

    def __str__(self):
        return '\n\n'.join([str(factor) for factor in self._factors])