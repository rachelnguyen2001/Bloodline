from factor import Factor
from bayes import BayesianNetwork


def create_bloodtype_cpt(var):
    maternal_gene = f'{var}_M'
    paternal_gene = f'{var}_P'
    blood_type = var
    probs = {('A', 'A', 'A'): 1.0, ('A', 'A', 'B'): 0.0, ('A', 'A', 'AB'): 0.0, ('A', 'A', 'O'): 0.0,
             ('A', 'B', 'A'): 0.0, ('A', 'B', 'B'): 0.0, ('A', 'B', 'AB'): 1.0, ('A', 'B', 'O'): 0.0,
             ('A', 'O', 'A'): 1.0, ('A', 'O', 'B'): 0.0, ('A', 'O', 'AB'): 0.0, ('A', 'O', 'O'): 0.0,
             ('B', 'A', 'A'): 0.0, ('B', 'A', 'B'): 0.0, ('B', 'A', 'AB'): 1.0, ('B', 'A', 'O'): 0.0,
             ('B', 'B', 'A'): 0.0, ('B', 'B', 'B'): 1.0, ('B', 'B', 'AB'): 0.0, ('B', 'B', 'O'): 0.0,
             ('B', 'O', 'A'): 0.0, ('B', 'O', 'B'): 1.0, ('B', 'O', 'AB'): 0.0, ('B', 'O', 'O'): 0.0,
             ('O', 'A', 'A'): 1.0, ('O', 'A', 'B'): 0.0, ('O', 'A', 'AB'): 0.0, ('O', 'A', 'O'): 0.0,
             ('O', 'B', 'A'): 0.0, ('O', 'B', 'B'): 1.0, ('O', 'B', 'AB'): 0.0, ('O', 'B', 'O'): 0.0,
             ('O', 'O', 'A'): 0.0, ('O', 'O', 'B'): 0.0, ('O', 'O', 'AB'): 0.0, ('O', 'O', 'O'): 1.0}
    return Factor([maternal_gene, paternal_gene, blood_type], probs)


def create_inheritance_cpt(parent, child, relation):
    maternal_gene = f'{parent}_M'
    paternal_gene = f'{parent}_P'
    inherited_gene = f'{child}_{relation}'
    probs = {('A', 'A', 'A'): 1.0, ('A', 'A', 'B'): 0.0, ('A', 'A', 'O'): 0.0,
             ('A', 'B', 'A'): 0.5, ('A', 'B', 'B'): 0.5, ('A', 'B', 'O'): 0.0,
             ('A', 'O', 'A'): 0.5, ('A', 'O', 'B'): 0.0, ('A', 'O', 'O'): 0.5,
             ('B', 'A', 'A'): 0.5, ('B', 'A', 'B'): 0.5, ('B', 'A', 'O'): 0.0,
             ('B', 'B', 'A'): 0.0, ('B', 'B', 'B'): 1.0, ('B', 'B', 'O'): 0.0,
             ('B', 'O', 'A'): 0.0, ('B', 'O', 'B'): 0.5, ('B', 'O', 'O'): 0.5,
             ('O', 'A', 'A'): 0.5, ('O', 'A', 'B'): 0.0, ('O', 'A', 'O'): 0.5,
             ('O', 'B', 'A'): 0.0, ('O', 'B', 'B'): 0.5, ('O', 'B', 'O'): 0.5,
             ('O', 'O', 'A'): 0.0, ('O', 'O', 'B'): 0.0, ('O', 'O', 'O'): 1.0}
    return Factor([maternal_gene, paternal_gene, inherited_gene], probs)


def create_inheritance_prior(person, relation):
    vars = [f'{person}_{relation}']
    probs = {('A',): 1/3, ('B',): 1/3, ('O',): 1/3}
    return Factor(vars, probs)


def create_vampire_bayes_net():
    domains = {'X_M': ['A', 'B', 'O'],
               'X_P': ['A', 'B', 'O'],
               'X': ['A', 'B', 'AB', 'O'],
               'Y_M': ['A', 'B', 'O'],
               'Y_P': ['A', 'B', 'O'],
               'Y': ['A', 'B', 'AB', 'O'],
               'Z_M': ['A', 'B', 'O'],
               'Z_P': ['A', 'B', 'O'],
               'Z': ['A', 'B', 'AB', 'O']}
    cpts = [create_bloodtype_cpt('X'), create_bloodtype_cpt('Y'), create_bloodtype_cpt('Z'),
            create_inheritance_cpt('X', 'Z', 'M'), create_inheritance_cpt('Y', 'Z', 'P'),
            create_inheritance_prior('X', 'M'), create_inheritance_prior('X', 'P'),
            create_inheritance_prior('X', 'M'), create_inheritance_prior('X', 'P')]
    return BayesianNetwork(cpts, domains)

