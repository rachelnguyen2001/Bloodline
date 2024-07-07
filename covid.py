from factor import Factor
from bayes import BayesianNetwork


def create_rapidtest_cpt(day):
    probs = {('-', '-'): 0.995, ('-', '+'): 0.005,
             ('+', '-'): 0.2, ('+', '+'): 0.8}
    return Factor([f'C_{day}', f'T_{day}'], probs)

def create_nextday_cpt(day):
    probs = {('-', '-'): 0.99, ('-', '+'): 0.01,
             ('+', '-'): 0.1, ('+', '+'): 0.9}
    return Factor([f'C_{day-1}', f'C_{day}'], probs)

def create_firstday_cpt():
    probs = {('-',): 0.99, ('+',): 0.01}
    return Factor([f'C_0'], probs)

def create_covid_bayes_net(last_day):
    covid_vars = [(f'C_{day}', ['-', '+']) for day in range(1, last_day+1)]
    rapidtest_vars = [(f'T_{day}', ['-', '+']) for day in range(1, last_day + 1)]
    all_vars = covid_vars + rapidtest_vars + [('C_0', ['-', '+'])]
    domains = {var: domain for var, domain in all_vars}
    cpts = ([create_rapidtest_cpt(day) for day in range(1, last_day + 1)] +
            [create_nextday_cpt(day) for day in range(1, last_day + 1)] +
            [create_firstday_cpt()])
    return BayesianNetwork(cpts, domains)
