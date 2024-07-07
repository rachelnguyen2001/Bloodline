from collections import defaultdict
from xml import dom


class Factor:
    """A factor in a Bayesian network (i.e. a multivariable function)"""

    def __init__(self, variables, values):
        """
        Parameters
        ----------
        variables : list[str]
            The variables of the factor
        values : dict[tuple[str], float]
            A dictionary mapping each event (expressed as a tuple) to its value
        """

        self._variables = variables
        self._values = values
    
    def get_variables(self):
        """Returns the variables of the factor.
        Returns
        -------
        list[Variable]
            The variables of the factor.
        """

        return self._variables

    def get_value(self, event):
        """Returns the value that the factor assigns to a particular event.
        Returns
        -------
        float
            The value associated with the event
        Raises
        ------
        KeyError
            If the factor has no value assigned to the given event.
        """

        key = []
        for var in self._variables:
            if var not in event:
                raise KeyError(f'Variable {var} not found in given event.')
            key.append(event[var])
        if tuple(key) in self._values:
            return self._values[tuple(key)]
        else:
            raise KeyError(f'No value assigned to event {event}.')

    def normalize(self):
        """Normalizes the event values.
        In other words, each event value is divided by the overall sum of the event
        values so that they all sum to one.
        Returns
        -------
        Factor
            A new Factor, identical to the current Factor, except that the event values
            are normalized.
        """
        # question two
        #get the sum of all possible values
        total_val = 0
        for i in self._values:
            total_val += self._values[i]
        new_values = {}
        #normalize the values
        for i in self._values:
            normalized_val = (self._values[i] / total_val)
            new_values[i] = normalized_val

        return Factor(self._variables, new_values)    

    def reduce(self, evidence):
        """Removes any events in the factor that do not agree with the "evidence" event.
        An event "does not agree" with another event if the two events associate different
        domain values with some variable. For instance, the following events agree:
            {'P': 'yes', 'D': 's', 'R': '+'}
            {'P': 'yes', 'D': 's', 'T': '-'}
        because there is no variable associated with different values in the two events.
        However:
            {'P': 'yes', 'D': 'n', 'R': '+'}
            {'P': 'yes', 'D': 's', 'T': '-'}
        do not agree, since the variable 'D' is associated with different values in the
        events.
        Parameters
        ----------
        evidence : dict[str, str]
            The "evidence" event.
        Returns
        -------
        Factor
            A new Factor, identical to the current Factor, except that events that disagree
            with the evidence event are removed.
        """
        # question two
        new_values = {}
        for i in self._values:
            flag = True
            for j in evidence:
                index = self._variables.index(j)
                if evidence[j] != i[index]:
                    flag = False
                    break 
            if flag:
                # keep the event that does not disagree with the "evidence" event
                new_values[i] = self._values[i]                                      
        return Factor(self._variables, new_values)

    def marginalize(self, variable):
        """Marginalizes (sums) out the specified variable.
        Parameters
        ----------
        variable : Variable
            The variable to marginalize out.
        Returns
        -------
        Factor
            A new Factor, identical to the current Factor with the specified variable
            marginalized out.
        """
        # question two
        new_variables = []
        # remove the marginalized variable
        for i in range(len(self._variables)):
            if self._variables[i] != variable:
                new_variables.append(self._variables[i])
        
        new_values = {}
        index = self._variables.index(variable)
        
        for i in self._values: 
            new_key = i[:index] + i[index+1:]
            if new_key not in new_values:
                # keep values of events that do not include the marginalized variable the same
                new_values[new_key] = self._values[i]
            else:
                # sums out the specified variable
                new_values[new_key] += self._values[i]
        return Factor(new_variables, new_values)

    def __str__(self):
        result = f"{self._variables}:"
        for event, value in self._values.items():
            result += f"\n  {event}: {value}"
        return result

    __repr__ = __str__

# recursive method that helps to build the list for events
def events_helper(i, vars, domains, cur_list):
    if i == len(vars):
        return cur_list
    var = domains[vars[i]]
    toAdd = []
    if len(cur_list) == 0:
        for val in var:
            new_dict = {vars[i]: val}
            toAdd.append(new_dict)
    else:
        for d in cur_list:
            # add the current val for var to all tuples in the list of events so far
            for val in var:
                d_dup = {}
                for v in d:
                    d_dup[v] = d[v]
                d_dup[vars[i]] = val

                toAdd.append(d_dup)
                
    return events_helper(i+1, vars, domains, toAdd)                
    
def events(vars, domains):
    """
    Takes a list of variables and returns the cross-product of the domains.
    For instance, suppose the domain of variable X is ('a', 'b') and the
    domain of the variable Y is ('c','d','e'). Then:
       >>> X = Variable('X', ('a', 'b'))
       >>> Y = Variable('Y', ('c', 'd', 'e'))
       >>> events([X, Y])
       [('a', 'c'), ('a', 'd'), ('a', 'e'), ('b', 'c'), ('b', 'd'), ('b', 'e')]
    """
    ... # question one
    returned_events = []
    returned_events = events_helper(0, vars, domains, [])
    return returned_events  

def multiply_factors(factors, domains):
    """Multiplies a list of factors.
    Parameters
    ----------
    factors : list[Factor]
        The factors to multiply
    domains : dict[str, list[str]]
        A dictionary mapping each variable to its possible values
    Returns
    -------
    Factor
        The product of the input factors.
    """
    # question three
    new_variables = []
    # unionize the variables of all the factors
    for f in factors:
        for v in f._variables:
            if v not in new_variables:
                new_variables.append(v)

    new_values = {}
    list_events = events(new_variables, domains)
    
    for i in list_events:
        val = 1
        for f in factors:            
            k = []
            for var in f._variables:
                # get the value for each variable in the current factor in event i
                k.append(i[var])
            
            new_key = tuple(k)
            if new_key in f._values:
                # multiply by the value associated with with event i in the current factor
                val *= f._values[new_key]
            else: 
                val = 0
        #assign the value to the dictionary "new_values"
        values = tuple(i.values())
        new_values[values] = val

    return Factor(new_variables, new_values)    