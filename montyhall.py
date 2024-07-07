from factor import Factor
from bayes import BayesianNetwork

def create_car_cpt():
    vars = ['C']
    probs = {('1',): 1/3, ('2',): 1/3, ('3',): 1/3}
    return Factor(vars, probs)


def create_goat_cpt():
    vars = ['C', 'G']
    probs = {('1', '2'): 0.5, ('1', '3',): 0.5,
             ('2', '2'): 0, ('2', '3',): 1,
             ('3', '2'): 1, ('3', '3',): 0}
    return Factor(vars, probs)


def create_finalchoice_cpt():
    vars = ['G', 'F']
    probs = {('2', '1'): 0, ('2', '2'): 0, ('2', '3',): 1,
             ('3', '1'): 0, ('3', '2'): 1, ('3', '3',): 0}
    return Factor(vars, probs)


def create_win_cpt():
    vars = ['F', 'C', 'W']
    probs = {('1', '1', 'yes'): 1, ('1', '1', 'no'): 0,
             ('1', '2', 'yes'): 0, ('1', '2', 'no'): 1,
             ('1', '3', 'yes'): 0, ('1', '3', 'no'): 1,
             ('2', '1', 'yes'): 0, ('2', '1', 'no'): 1,
             ('2', '2', 'yes'): 1, ('2', '2', 'no'): 0,
             ('2', '3', 'yes'): 0, ('2', '3', 'no'): 1,
             ('3', '1', 'yes'): 0, ('3', '1', 'no'): 1,
             ('3', '2', 'yes'): 0, ('3', '2', 'no'): 1,
             ('3', '3', 'yes'): 1, ('3', '3', 'no'): 0}
    return Factor(vars, probs)


def create_montyhall_bayes_net():
    domains = {'C': ['1', '2', '3'],
               'G': ['2', '3'],
               'F': ['1', '2', '3'],
               'W': ['yes', 'no']}
    cpts = [create_car_cpt(),
            create_goat_cpt(),
            create_finalchoice_cpt(),
            create_win_cpt()]
    return BayesianNetwork(cpts, domains)
