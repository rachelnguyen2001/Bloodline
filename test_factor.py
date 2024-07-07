import unittest
import pandas as pd
from factor import Factor, multiply_factors, events
from montyhall import create_goat_cpt, create_finalchoice_cpt
from vampire import create_inheritance_cpt

class TestOne(unittest.TestCase):

    def test_events1(self):
        domains = {'P': ['yes', 'no'],
                   'D': ['n', 's', 'e', 'w'],
                   'R': ['+', '-']}
        computed = events(['P', 'D'], domains)
        expected = [{'P': 'yes', 'D': 'n'},
                    {'P': 'yes', 'D': 's'},
                    {'P': 'yes', 'D': 'e'},
                    {'P': 'yes', 'D': 'w'},
                    {'P': 'no', 'D': 'n'},
                    {'P': 'no', 'D': 's'},
                    {'P': 'no', 'D': 'e'},
                    {'P': 'no', 'D': 'w'}]
        for event in computed:
            self.assertIn(event, expected)
        for event in expected:
            self.assertIn(event, computed)

    def test_events2(self):
        domains = {'P': ['yes', 'no'],
                   'D': ['n', 's', 'e', 'w'],
                   'R': ['+', '-']}
        computed = events(['P', 'D', 'R'], domains)
        expected = [{'P': 'yes', 'D': 'n', 'R': '+'},
                    {'P': 'yes', 'D': 'n', 'R': '-'},
                    {'P': 'yes', 'D': 's', 'R': '+'},
                    {'P': 'yes', 'D': 's', 'R': '-'},
                    {'P': 'yes', 'D': 'e', 'R': '+'},
                    {'P': 'yes', 'D': 'e', 'R': '-'},
                    {'P': 'yes', 'D': 'w', 'R': '+'},
                    {'P': 'yes', 'D': 'w', 'R': '-'},
                    {'P': 'no', 'D': 'n', 'R': '+'},
                    {'P': 'no', 'D': 'n', 'R': '-'},
                    {'P': 'no', 'D': 's', 'R': '+'},
                    {'P': 'no', 'D': 's', 'R': '-'},
                    {'P': 'no', 'D': 'e', 'R': '+'},
                    {'P': 'no', 'D': 'e', 'R': '-'},
                    {'P': 'no', 'D': 'w', 'R': '+'},
                    {'P': 'no', 'D': 'w', 'R': '-'}]
        for event in computed:
            self.assertIn(event, expected)
        for event in expected:
            self.assertIn(event, computed)



def example_factors():
    domains = {'P': ['yes', 'no'], 'L': ['u', 'd']}
    p_factor = Factor(['P'], {
        ('yes',): 0.87,
        ('no',): 0.13})
    l_factor = Factor(['P', 'L'], {
        ('yes', 'u'): 0.1,
        ('yes', 'd'): 0.9,
        ('no', 'u'): 0.99,
        ('no', 'd'): 0.01})
    return domains, p_factor, l_factor


class TestTwoNormalizeAndReduce(unittest.TestCase):

    def test_normalize1(self):
        _, _, l_factor = example_factors()
        l_factor = l_factor.normalize()
        self.assertEqual(l_factor.get_value({'P': 'yes', 'L': 'u'}), .05)
        self.assertEqual(l_factor.get_value({'P': 'yes', 'L': 'd'}), .45)
        self.assertEqual(l_factor.get_value({'P': 'no', 'L': 'u'}), .495)
        self.assertEqual(l_factor.get_value({'P': 'no', 'L': 'd'}), .005)

    def test_normalize2(self):
        factor = create_inheritance_cpt('X', 'Z', 'M').normalize()
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'A', 'Z_M': 'A'}), 1/9)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'A', 'Z_M': 'B'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'A', 'Z_M': 'O'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'B', 'Z_M': 'A'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'B', 'Z_M': 'B'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'B', 'Z_M': 'O'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'O', 'Z_M': 'A'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'O', 'Z_M': 'B'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'A', 'X_P': 'O', 'Z_M': 'O'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'A', 'Z_M': 'A'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'A', 'Z_M': 'B'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'A', 'Z_M': 'O'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'B', 'Z_M': 'A'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'B', 'Z_M': 'B'}), 1/9)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'B', 'Z_M': 'O'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'O', 'Z_M': 'A'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'O', 'Z_M': 'B'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'B', 'X_P': 'O', 'Z_M': 'O'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'A', 'Z_M': 'A'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'A', 'Z_M': 'B'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'A', 'Z_M': 'O'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'B', 'Z_M': 'A'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'B', 'Z_M': 'B'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'B', 'Z_M': 'O'}), 1/18)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'O', 'Z_M': 'A'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'O', 'Z_M': 'B'}), 0.0)
        self.assertEqual(factor.get_value({'X_M': 'O', 'X_P': 'O', 'Z_M': 'O'}), 1/9)

    def test_reduce(self):
        _, _, l_factor = example_factors()
        l_factor = l_factor.reduce({'P': 'yes'})
        self.assertEqual(l_factor.get_value({'P': 'yes', 'L': 'u'}), .1)
        self.assertEqual(l_factor.get_value({'P': 'yes', 'L': 'd'}), .9)
        with self.assertRaises(KeyError):
            l_factor.get_value({'P': 'no', 'L': 'u'})
        with self.assertRaises(KeyError):
            l_factor.get_value({'P': 'no', 'L': 'd'})


class TestTwoMarginalize(unittest.TestCase):

    def test_marginalize(self):
        _, _, l_factor = example_factors()
        factor = l_factor.marginalize('L')
        self.assertEqual(factor.get_value({'P': 'yes'}), 1.0)
        self.assertEqual(factor.get_value({'P': 'no'}), 1.0)
        factor = l_factor.marginalize('P')
        self.assertEqual(factor.get_value({'L': 'u'}), 1.09)
        self.assertEqual(factor.get_value({'L': 'd'}), .91)

    def test_marginalize2(self):
        factor = create_inheritance_cpt('X', 'Z', 'M').marginalize('X_P')
        self.assertEqual(factor.get_value({'X_M': 'A', 'Z_M': 'A'}), 2.0)
        self.assertEqual(factor.get_value({'X_M': 'A', 'Z_M': 'B'}), 0.5)
        self.assertEqual(factor.get_value({'X_M': 'A', 'Z_M': 'O'}), 0.5)
        self.assertEqual(factor.get_value({'X_M': 'B', 'Z_M': 'A'}), 0.5)
        self.assertEqual(factor.get_value({'X_M': 'B', 'Z_M': 'B'}), 2.0)
        self.assertEqual(factor.get_value({'X_M': 'B', 'Z_M': 'O'}), 0.5)
        self.assertEqual(factor.get_value({'X_M': 'O', 'Z_M': 'A'}), 0.5)
        self.assertEqual(factor.get_value({'X_M': 'O', 'Z_M': 'B'}), 0.5)
        self.assertEqual(factor.get_value({'X_M': 'O', 'Z_M': 'O'}), 2.0)


class TestThree(unittest.TestCase):

    def test_multiply1(self):
        domains, p_factor, l_factor = example_factors()
        product = multiply_factors([p_factor, l_factor], domains)
        self.assertAlmostEqual(product.get_value({'P': 'yes', 'L': 'u'}), .087)
        self.assertAlmostEqual(product.get_value({'P': 'yes', 'L': 'd'}), .783)
        self.assertAlmostEqual(product.get_value({'P': 'no', 'L': 'u'}), .1287)
        self.assertAlmostEqual(product.get_value({'P': 'no', 'L': 'd'}), .0013)

    def test_multiply2(self):
        domains = {'C': ['1', '2', '3'],
                   'G': ['2', '3'],
                   'F': ['1', '2', '3'],
                   'W': ['yes', 'no']}
        factor1 = create_goat_cpt()
        factor2 = create_finalchoice_cpt()
        product = multiply_factors([factor1, factor2], domains)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '2', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '2', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '2', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '3', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '3', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '3', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '2', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '2', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '2', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '3', 'C': '1'}), 0.5)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '3', 'C': '2'}), 1.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '3', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '2', 'C': '1'}), 0.5)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '2', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '2', 'C': '3'}), 1.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '3', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '3', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '3', 'C': '3'}), 0.0)

    def test_multiply3(self):
        domains = {'C': ['1', '2', '3'],
                        'G': ['2', '3'],
                        'F': ['1', '2', '3'],
                        'W': ['yes', 'no']}
        factor1 = create_goat_cpt().reduce({'G': '3'})
        factor2 = create_finalchoice_cpt()
        product = multiply_factors([factor1, factor2], domains)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '2', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '2', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '2', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '3', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '3', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '1', 'G': '3', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '2', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '2', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '2', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '3', 'C': '1'}), 0.5)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '3', 'C': '2'}), 1.0)
        self.assertAlmostEqual(product.get_value({'F': '2', 'G': '3', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '2', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '2', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '2', 'C': '3'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '3', 'C': '1'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '3', 'C': '2'}), 0.0)
        self.assertAlmostEqual(product.get_value({'F': '3', 'G': '3', 'C': '3'}), 0.0)

class TestFactor(unittest.TestCase):

    def test_get_variables(self):
        _, p_factor, l_factor = example_factors()
        self.assertEqual(p_factor.get_variables(), ['P'])
        self.assertEqual(set(l_factor.get_variables()), set(['P', 'L']))

    def test_get_value(self):
        _, _, l_factor = example_factors()
        self.assertEqual(l_factor.get_value({'P': 'yes', 'L': 'u'}), .1)
        self.assertEqual(l_factor.get_value({'P': 'yes', 'L': 'd'}), .9)
        self.assertEqual(l_factor.get_value({'P': 'no', 'L': 'u'}), .99)
        self.assertEqual(l_factor.get_value({'P': 'no', 'L': 'd'}), .01)
        with self.assertRaises(KeyError):
            l_factor.get_value({'P': 'yes', 'M': 'u'})
        with self.assertRaises(KeyError):
            l_factor.get_value({'P': 'yeah', 'L': 'u'})




if __name__ == "__main__":
    unittest.main()   