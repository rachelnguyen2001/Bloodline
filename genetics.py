import numpy as np
import os
from factor import events, Factor
from bayes import BayesianNetwork


class FamilyMember:
    """A single member of a family tree."""

    def __init__(self, name, sex, mother, father):
        """
        Parameters
        ----------
        name : str
            The name of the family member.
        sex : str
            The sex of the family member ("male" or "female")
        mother : FamilyMember
            The mother of the family member (or None if unknown)
        father : FamilyMember
            The father of the family member (or None if unknown)
        """

        self.name = name
        self.sex = sex
        self.mother = mother
        self.father = father

    def get_name(self):
        """Returns the name of the family member."""
        return self.name

    def get_sex(self):
        """Returns the sex of the family member."""
        return self.sex


class Male(FamilyMember):
    """A male family member."""

    def __init__(self, name, mother=None, father=None):
        super().__init__(name, "male", mother, father)


class Female(FamilyMember):
    """A female family member."""

    def __init__(self, name, mother=None, father=None):
        super().__init__(name, "female", mother, father)


def romanoffs():
    """A simple example of a family, using four members of the Russian royal family (the Romanoffs)."""
    alexandra = Female("alexandra")
    nicholas = Male("nicholas")
    alexey = Male("alexey", mother=alexandra, father=nicholas)
    anastasia = Female("anastasia", mother=alexandra, father=nicholas)
    return alexandra, nicholas, alexey, anastasia


def create_variable_domains(family):
    """Creates a dictionary mapping each variable to its domain, for the hemophilia network.
    For each family member, we create either 3 or 4 variables (3 if they’re male, 4 if they’re female).
    If N is the name of the family member, then we create the following variables:
        M_N: N’s maternally inherited gene
        P_N: N’s paternally inherited gene (if N is female)
        G_N: the genotype of N
        H_N: whether N has hemophilia
    The variables should be mapped to the following domains:
        - M_N: ['x', 'X']
        - P_N: ['x', 'X']
        - G_N: ['xx', 'xX', 'XX']
        - H_N: ['-', '+']
    Parameters
    ----------
    family : list[FamilyMember]
        the list of family members
    Returns
    -------
    dict[str, list[str]]
        a dictionary mapping each variable to its domain (i.e. its possible values)
    """
    # question five
    domains = {}   
     
    #hardcode the domains
    for member in family:
        name = member.name
        domains["M_" + name] = ['x','X']
        domains["H_" + name] = ['-', '+']
        if member.sex == "male":
            domains["G_" + name] = ['xy', 'Xy']
        else:
            domains["G_" + name] = ['xx', 'xX', 'XX']
            domains["P_" + name] = ['x', 'X']
    
    return domains


def create_hemophilia_cpt(person):
    """Creates a conditional probability table (CPT) specifying the probability of hemophilia, given one's genotype.
    Parameters
    ----------
    person : FamilyMember
        the family member whom the CPT pertains to
    Returns
    -------
    Factor
        a Factor specifying the probability of hemophilia, given one's genotype
    """
    # question six
    G = "G_" + person.name
    H = "H_" + person.name
    variables = [G, H]
    values = {}
    domains = create_variable_domains([person])
    event_list = events(variables, domains)

    for event in event_list:
        if person.sex == 'male':
            if event[H] == '+':
                if "X" in event[G]:
                    #Xy is going to have H
                    values[tuple(event.values())] = 1.0
                else: 
                    #"xy" would not have H
                    values[tuple(event.values())] = 0.0
            if event[H] == '-':
                if "X" in event[G]:
                    values[tuple(event.values())] = 0.0
                else:
                    values[tuple(event.values())] = 1.0
        if person.sex == 'female':
            if event[H] == '+':
                if "XX" in event[G]:
                    #only XX would have H
                    values[tuple(event.values())] = 1.0
                else: 
                    values[tuple(event.values())] = 0.0
            if event[H] == '-':
                if "XX" in event[G]:
                    values[tuple(event.values())] = 0.0
                else: 
                    values[tuple(event.values())] = 1.0

    return Factor(variables, values)



def create_genotype_cpt(person):
    """Creates a conditional probability table (CPT) specifying the probability of a genotype, given one's inherited genes.
    Parameters
    ----------
    person : FamilyMember
        the family member whom the CPT pertains to
    Returns
    -------
    Factor
        a Factor specifying the probability of a genotype, given one's inherited genes
    """
    # question seven
    P = "P_" + person.name
    M = "M_" + person.name
    G = "G_" + person.name
    variables = []
    values = {}
    domains = create_variable_domains([person])
    
    #separate by sex of the person. If male we do not care about paternal as it is always "y"
    if person.sex == 'male':
        variables = [M, G]
        event_list = events(variables, domains)
        
        for event in event_list:
            if event[M] in event[G]:
                values[tuple(event.values())] = 1.0
            else:
                values[tuple(event.values())] = 0.0
    else:
        variables = [P, M, G]
        event_list = events(variables, domains)
        for event in event_list:
            # compare whether the genotype inherited matches the person's genotype
            # if so, assign 1.0, else assign 0.0
            genotype = ''.join(sorted(event[P] + event[M]))
            sorted_G = ''.join(sorted(event[G]))
            if sorted_G == genotype:
                values[tuple(event.values())] = 1.0
            else:
                values[tuple(event.values())] = 0.0

    return Factor(variables, values)

def create_maternal_inheritance_cpt(person):
    """Creates a conditional probability table (CPT) specifying the probability of the gene inherited from one's mother.
    Parameters
    ----------
    person : FamilyMember
        the family member whom the CPT pertains to
    Returns
    -------
    Factor
        a Factor specifying the probability of the gene inherited from the family member's mother.
    """
    # question eight
    M = "M_" + person.name
    variables = [M]
    values = {}
    mother = person.mother
    if mother == None:
        values[tuple('x')] = 29999/30000
        values[tuple('X')] = 1/30000
    else:
        G = "G_" + mother.name
        variables.append(G)
        domains = create_variable_domains([person, mother])
        event_list = events(variables, domains)

        for event in event_list:
            if event[G] == 'xx':
                if event[M] == 'x':
                    values[tuple(event.values())] = 1.0
                else:
                    values[tuple(event.values())] = 0.0
            elif event[G] == 'XX':
                if event[M] == 'x':
                    values[tuple(event.values())] = 0.0
                else:
                    values[tuple(event.values())] = 1.0
            else:
                values[tuple(event.values())] = 0.5
    return Factor(variables, values)


def create_paternal_inheritance_cpt(person):
    """Creates a conditional probability table (CPT) specifying the probability of the gene inherited from one's father.
    Parameters
    ----------
    person : FamilyMember
        the family member whom the CPT pertains to
    Returns
    -------
    Factor
        a Factor specifying the probability of the gene inherited from the family member's father.
    """
    # question nine
    P = "P_" + person.name
    variables = [P]
    values = {}
    if person.sex == 'male':
        values[tuple('y')] = 1
    else:
        father = person.father
        if father is None:
            values[tuple('x')] = 29999/30000
            values[tuple('X')] = 1/30000
        else:
            G = "G_" + father.name
            variables.append(G)
            domains = create_variable_domains([person, father])
            event_list = events(variables, domains)
            
            for event in event_list:
                if event[P] in event[G]:
                    values[tuple(event.values())] = 1.0
                else:
                    values[tuple(event.values())] = 0.0

    return Factor(variables, values)  

def create_family_bayes_net(family):
    """Creates a Bayesian network that models the genetic inheritance of hemophilia within a family.
    Parameters
    ----------
    family : list[FamilyMember]
        the members of the family
    Returns
    -------
    BayesianNetwork
        a Bayesian network that models the genetic inheritance of hemophilia within the specified family
    """
    domains = create_variable_domains(family)
    cpts = []
    for person in family:
        if person.get_sex() == "female":
            cpts.append(create_paternal_inheritance_cpt(person))
        cpts.append(create_maternal_inheritance_cpt(person))
        cpts.append(create_genotype_cpt(person))
        cpts.append(create_hemophilia_cpt(person))
    return BayesianNetwork(cpts, domains)