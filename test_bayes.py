import unittest
import pandas as pd
from montyhall import create_montyhall_bayes_net
from vampire import create_vampire_bayes_net


def compute_probability(bnet, event):
    return bnet.compute_marginal(event.keys()).get_value(event)

def compute_conditional_probability(bnet, event, evidence):
    return bnet.compute_conditional(event.keys(), evidence).get_value(event)

class TestFour(unittest.TestCase):

    def test_variable_elimination1(self):
        prob = compute_probability(create_montyhall_bayes_net(),
                                   {'W': 'yes'})
        self.assertAlmostEqual(prob, 2/3)

    def test_variable_elimination2(self):
        prob = compute_probability(create_montyhall_bayes_net(),
                                   {'W': 'yes', 'G': '2'})
        self.assertAlmostEqual(prob, 1/3)

    def test_variable_elimination3(self):
        prob = compute_probability(create_vampire_bayes_net(),
                                   {'Z': 'AB'})
        self.assertAlmostEqual(prob, 2/9)

    def test_variable_elimination4(self):
        prob = compute_probability(create_vampire_bayes_net(),
                                   {'Z': 'AB', 'X': 'O'})
        self.assertAlmostEqual(prob, 0)

    def test_variable_elimination5(self):
        prob = compute_probability(create_vampire_bayes_net(),
                                   {'Z': 'AB', 'X': 'A', 'Y': 'B'})
        self.assertAlmostEqual(prob, 0.04938271604938271)

    def test_conditional_probability1(self):
        prob = compute_conditional_probability(create_montyhall_bayes_net(),
                                               {'W': 'yes'},
                                               {'G': '2'})
        self.assertAlmostEqual(prob, 2/3)

    def test_conditional_probability2(self):
        prob = compute_conditional_probability(create_vampire_bayes_net(),
                                               {'Y': 'AB'},
                                               {'Z': 'A'})
        self.assertAlmostEqual(prob, 2/9)

    def test_conditional_probability3(self):
        prob = compute_conditional_probability(create_vampire_bayes_net(),
                                               {'Y': 'AB'},
                                               {'Z': 'O'})
        self.assertAlmostEqual(prob, 0)

    def test_conditional_probability4(self):
        prob = compute_conditional_probability(create_vampire_bayes_net(),
                                               {'Y': 'B'},
                                               {'Z': 'AB', 'X': 'A'})
        self.assertAlmostEqual(prob, 2/3)

    def test_conditional_probability5(self):
        prob = compute_conditional_probability(create_vampire_bayes_net(),
                                               {'Y': 'AB'},
                                               {'Z': 'AB', 'X': 'A'})
        self.assertAlmostEqual(prob, 1/3)

if __name__ == "__main__":
    unittest.main()   