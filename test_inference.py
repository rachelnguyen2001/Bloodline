import unittest
from montyhall import create_montyhall_bayes_net
from covid import create_covid_bayes_net
from vampire import create_vampire_bayes_net
from util import UndirectedGraph, build_junction_tree
from inference import count_nodes, compute_separators, message_passing
from inference import belief_propagation

def compute_probability(bnet, event):
    return bnet.compute_marginal(event.keys()).get_value(event)

def compute_conditional_probability(bnet, event, evidence):
    return bnet.compute_conditional(event.keys(), evidence).get_value(event)

class TestMessagePassing(unittest.TestCase):

    def test_message_passing1(self):
        def compute_leaf_message(leaf):
            msgs = {1: {'C', 'E'}, 4: {'B', 'D'} , 5: {'A', 'B'} , 6: {'B', 'C'} , 7: {'A'}}
            return msgs[leaf]
        tree = UndirectedGraph(8, [(0, 2), (0, 3), (0, 4), (1, 2), (2, 6), (3, 5), (3, 7)])
        messages = message_passing(tree, compute_leaf_message, lambda s, d, msgs: set.union(*msgs))
        expected = {(1, 2): {'C', 'E'},
                    (4, 0): {'B', 'D'},
                    (5, 3): {'B', 'A'},
                    (6, 2): {'C', 'B'},
                    (7, 3): {'A'},
                    (2, 0): {'C', 'E', 'B'},
                    (3, 0): {'B', 'A'},
                    (0, 2): {'B', 'A', 'D'},
                    (0, 4): {'C', 'E', 'B', 'A'},
                    (2, 1): {'C', 'B', 'A', 'D'},
                    (2, 6): {'C', 'E', 'B', 'A', 'D'},
                    (0, 3): {'C', 'E', 'B', 'D'},
                    (3, 5): {'C', 'E', 'B', 'A', 'D'},
                    (3, 7): {'C', 'E', 'B', 'A', 'D'}}
        self.assertEqual(messages, expected)

    def test_message_passing2(self):
        bnet = create_montyhall_bayes_net()
        jtree = build_junction_tree(bnet)
        self.assertEqual(count_nodes(jtree), 6)


class TestTen(unittest.TestCase):

    def test_get_separator(self):
        bnet = create_covid_bayes_net(2)
        jtree = build_junction_tree(bnet)
        separators = compute_separators(jtree)
        self.assertEqual(separators[(1,2)], {'C_2'})
        self.assertEqual(separators[(2,6)], {'C_1', 'C_2'})
        self.assertEqual(separators[(0,2)], {'C_1'})
        self.assertEqual(separators[(0,3)], {'C_1'})
        self.assertEqual(separators[(0,4)], {'C_1'})
        self.assertEqual(separators[(3,5)], {'C_0', 'C_1'})
        self.assertEqual(separators[(3,7)], {'C_0'})

    def test_get_separator2(self):
        bnet = create_vampire_bayes_net()
        jtree = build_junction_tree(bnet)
        separators = compute_separators(jtree)
        self.assertEqual(separators[(0, 1)], {'X_M', 'X_P'})
        self.assertEqual(separators[(1, 0)], {'X_M', 'X_P'})
        self.assertEqual(separators[(0, 8)], {'X_M', 'X_P'})
        self.assertEqual(separators[(8, 0)], {'X_M', 'X_P'})
        self.assertEqual(separators[(0, 9)], {'X_M'})
        self.assertEqual(separators[(9, 0)], {'X_M'})
        self.assertEqual(separators[(0, 10)], {'X_P'})
        self.assertEqual(separators[(10, 0)], {'X_P'})
        self.assertEqual(separators[(0, 11)], {'X_M'})
        self.assertEqual(separators[(11, 0)], {'X_M'})
        self.assertEqual(separators[(0, 12)], {'X_P'})
        self.assertEqual(separators[(12, 0)], {'X_P'})
        self.assertEqual(separators[(1, 4)], {'Z_M'})
        self.assertEqual(separators[(4, 1)], {'Z_M'})
        self.assertEqual(separators[(1, 6)], {'Z_M', 'X_M', 'X_P'})
        self.assertEqual(separators[(6, 1)], {'Z_M', 'X_M', 'X_P'})
        self.assertEqual(separators[(2, 3)], {'Y_M', 'Y_P'})
        self.assertEqual(separators[(3, 2)], {'Y_M', 'Y_P'})
        self.assertEqual(separators[(3, 4)], {'Z_P'})
        self.assertEqual(separators[(4, 3)], {'Z_P'})
        self.assertEqual(separators[(3, 7)], {'Y_M', 'Y_P', 'Z_P'})
        self.assertEqual(separators[(7, 3)], {'Y_M', 'Y_P', 'Z_P'})
        self.assertEqual(separators[(4, 5)], {'Z_P', 'Z_M'})
        self.assertEqual(separators[(5, 4)], {'Z_P', 'Z_M'})


class TestEleven(unittest.TestCase):

    def test_bp1(self):
        marginals = belief_propagation(create_vampire_bayes_net(),
                                       {'Z': 'AB', 'X': 'A'})
        self.assertAlmostEqual(marginals['X_M'].get_value({'X_M': 'A'}), 0.75)
        self.assertAlmostEqual(marginals['X_M'].get_value({'X_M': 'B'}), 0.0)
        self.assertAlmostEqual(marginals['X_M'].get_value({'X_M': 'O'}), 0.25)
        self.assertAlmostEqual(marginals['Y_M'].get_value({'Y_M': 'A'}), 1/6)
        self.assertAlmostEqual(marginals['Y_M'].get_value({'Y_M': 'B'}), 2/3)
        self.assertAlmostEqual(marginals['Y_M'].get_value({'Y_M': 'O'}), 1/6)
        self.assertAlmostEqual(marginals['Z_M'].get_value({'Z_M': 'A'}), 1.0)
        self.assertAlmostEqual(marginals['Z_M'].get_value({'Z_M': 'B'}), 0.0)
        self.assertAlmostEqual(marginals['Z_M'].get_value({'Z_M': 'O'}), 0.0)
        self.assertAlmostEqual(marginals['X_P'].get_value({'X_P': 'A'}), 0.75)
        self.assertAlmostEqual(marginals['X_P'].get_value({'X_P': 'B'}), 0.0)
        self.assertAlmostEqual(marginals['X_P'].get_value({'X_P': 'O'}), 0.25)
        self.assertAlmostEqual(marginals['Y_P'].get_value({'Y_P': 'A'}), 1/6)
        self.assertAlmostEqual(marginals['Y_P'].get_value({'Y_P': 'B'}), 2/3)
        self.assertAlmostEqual(marginals['Y_P'].get_value({'Y_P': 'O'}), 1/6)
        self.assertAlmostEqual(marginals['Z_P'].get_value({'Z_P': 'A'}), 0.0)
        self.assertAlmostEqual(marginals['Z_P'].get_value({'Z_P': 'B'}), 1.0)
        self.assertAlmostEqual(marginals['Z_P'].get_value({'Z_P': 'O'}), 0.0)
        self.assertAlmostEqual(marginals['X'].get_value({'X': 'A'}), 1.0)
        self.assertAlmostEqual(marginals['Y'].get_value({'Y': 'A'}), 0.0)
        self.assertAlmostEqual(marginals['Y'].get_value({'Y': 'B'}), 2/3)
        self.assertAlmostEqual(marginals['Y'].get_value({'Y': 'AB'}), 1/3)
        self.assertAlmostEqual(marginals['Y'].get_value({'Y': 'O'}), 0.0)
        self.assertAlmostEqual(marginals['Z'].get_value({'Z': 'AB'}), 1.0)


    def test_bp2(self):
        marginals = belief_propagation(create_covid_bayes_net(3),
                                       {'T_3': '+'})
        self.assertAlmostEqual(marginals['C_0'].get_value({'C_0': '-'}), 0.8162153033624054)
        self.assertAlmostEqual(marginals['C_0'].get_value({'C_0': '+'}), 0.1837846966375946)
        self.assertAlmostEqual(marginals['C_1'].get_value({'C_1': '-'}), 0.61536930712012)
        self.assertAlmostEqual(marginals['C_1'].get_value({'C_1': '+'}), 0.38463069287988005)
        self.assertAlmostEqual(marginals['C_2'].get_value({'C_2': '-'}), 0.39473140840288345)
        self.assertAlmostEqual(marginals['C_2'].get_value({'C_2': '+'}), 0.6052685915971165)
        self.assertAlmostEqual(marginals['C_3'].get_value({'C_3': '-'}), 0.15130192341914694)
        self.assertAlmostEqual(marginals['C_3'].get_value({'C_3': '+'}), 0.8486980765808531)
        self.assertAlmostEqual(marginals['T_1'].get_value({'T_1': '-'}), 0.6892185991604953)
        self.assertAlmostEqual(marginals['T_1'].get_value({'T_1': '+'}), 0.31078140083950473)
        self.assertAlmostEqual(marginals['T_2'].get_value({'T_2': '-'}), 0.5138114696802923)
        self.assertAlmostEqual(marginals['T_2'].get_value({'T_2': '+'}), 0.4861885303197077)
        self.assertAlmostEqual(marginals['T_3'].get_value({'T_3': '+'}), 1.0)



if __name__ == "__main__":
    unittest.main()   