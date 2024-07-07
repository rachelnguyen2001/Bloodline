import unittest
from genetics import create_family_bayes_net, romanoffs, create_variable_domains
from genetics import create_hemophilia_cpt, create_genotype_cpt
from genetics import create_maternal_inheritance_cpt, create_paternal_inheritance_cpt

def compute_conditional_probability(bnet, event, evidence):
    return bnet.compute_conditional(event.keys(), evidence).get_value(event)

class TestFive(unittest.TestCase):

    def test_create_variable_domains(self):
        expected = {'P_alexandra': ['x', 'X'], 'M_alexandra': ['x', 'X'],
                    'G_alexandra': ['xx', 'xX', 'XX'], 'H_alexandra': ['-', '+'],
                    'M_nicholas': ['x', 'X'], 'G_nicholas': ['xy', 'Xy'], 'H_nicholas': ['-', '+'],
                    'M_alexey': ['x', 'X'], 'G_alexey': ['xy', 'Xy'], 'H_alexey': ['-', '+'],
                    'P_anastasia': ['x', 'X'], 'M_anastasia': ['x', 'X'],
                    'G_anastasia': ['xx', 'xX', 'XX'], 'H_anastasia': ['-', '+']}
        self.assertEqual(create_variable_domains(romanoffs()), expected)

class TestSix(unittest.TestCase):

    def test_create_hemophilia_cpt_alexandra(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_hemophilia_cpt(alexandra)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xx', 'H_alexandra': '-'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xx', 'H_alexandra': '+'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xX', 'H_alexandra': '-'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xX', 'H_alexandra': '+'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'XX', 'H_alexandra': '-'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'XX', 'H_alexandra': '+'}), 1.0)

    def test_create_hemophilia_cpt_nicholas(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_hemophilia_cpt(nicholas)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'xy', 'H_nicholas': '-'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'xy', 'H_nicholas': '+'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'Xy', 'H_nicholas': '-'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'Xy', 'H_nicholas': '+'}), 1.0)

    def test_create_hemophilia_cpt_alexey(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_hemophilia_cpt(alexey)
        self.assertAlmostEqual(cpt.get_value({'G_alexey': 'xy', 'H_alexey': '-'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexey': 'xy', 'H_alexey': '+'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexey': 'Xy', 'H_alexey': '-'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexey': 'Xy', 'H_alexey': '+'}), 1.0)

    def test_create_hemophilia_cpt_anastasia(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_hemophilia_cpt(anastasia)
        self.assertAlmostEqual(cpt.get_value({'G_anastasia': 'xx', 'H_anastasia': '-'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_anastasia': 'xx', 'H_anastasia': '+'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_anastasia': 'xX', 'H_anastasia': '-'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_anastasia': 'xX', 'H_anastasia': '+'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_anastasia': 'XX', 'H_anastasia': '-'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_anastasia': 'XX', 'H_anastasia': '+'}), 1.0)


class TestSeven(unittest.TestCase):

    def test_create_cpt_alexandra(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_genotype_cpt(alexandra)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x', 'M_alexandra': 'x', 'G_alexandra': 'xx'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x', 'M_alexandra': 'x', 'G_alexandra': 'xX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x', 'M_alexandra': 'x', 'G_alexandra': 'XX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x', 'M_alexandra': 'X', 'G_alexandra': 'xx'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x', 'M_alexandra': 'X', 'G_alexandra': 'xX'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x', 'M_alexandra': 'X', 'G_alexandra': 'XX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X', 'M_alexandra': 'x', 'G_alexandra': 'xx'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X', 'M_alexandra': 'x', 'G_alexandra': 'xX'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X', 'M_alexandra': 'x', 'G_alexandra': 'XX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X', 'M_alexandra': 'X', 'G_alexandra': 'xx'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X', 'M_alexandra': 'X', 'G_alexandra': 'xX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X', 'M_alexandra': 'X', 'G_alexandra': 'XX'}), 1.0)

    def test_create_cpt_nicholas(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_genotype_cpt(nicholas)
        self.assertAlmostEqual(cpt.get_value({'M_nicholas': 'x', 'G_nicholas': 'xy'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'M_nicholas': 'x', 'G_nicholas': 'Xy'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'M_nicholas': 'X', 'G_nicholas': 'xy'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'M_nicholas': 'X', 'G_nicholas': 'Xy'}), 1.0)

    def test_create_cpt_alexey(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_genotype_cpt(alexey)
        self.assertAlmostEqual(cpt.get_value({'M_alexey': 'x', 'G_alexey': 'xy'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'M_alexey': 'x', 'G_alexey': 'Xy'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'M_alexey': 'X', 'G_alexey': 'xy'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'M_alexey': 'X', 'G_alexey': 'Xy'}), 1.0)

    def test_create_cpt_anastasia(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_genotype_cpt(anastasia)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'x', 'M_anastasia': 'x', 'G_anastasia': 'xx'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'x', 'M_anastasia': 'x', 'G_anastasia': 'xX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'x', 'M_anastasia': 'x', 'G_anastasia': 'XX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'x', 'M_anastasia': 'X', 'G_anastasia': 'xx'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'x', 'M_anastasia': 'X', 'G_anastasia': 'xX'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'x', 'M_anastasia': 'X', 'G_anastasia': 'XX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'X', 'M_anastasia': 'x', 'G_anastasia': 'xx'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'X', 'M_anastasia': 'x', 'G_anastasia': 'xX'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'X', 'M_anastasia': 'x', 'G_anastasia': 'XX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'X', 'M_anastasia': 'X', 'G_anastasia': 'xx'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'X', 'M_anastasia': 'X', 'G_anastasia': 'xX'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'P_anastasia': 'X', 'M_anastasia': 'X', 'G_anastasia': 'XX'}), 1.0)


class TestEight(unittest.TestCase):

    def test_create_cpt_alexandra(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_maternal_inheritance_cpt(alexandra)
        self.assertAlmostEqual(cpt.get_value({'M_alexandra': 'x'}), 29999/30000)
        self.assertAlmostEqual(cpt.get_value({'M_alexandra': 'X'}), 1/30000)

    def test_create_cpt_nicholas(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_maternal_inheritance_cpt(nicholas)
        self.assertAlmostEqual(cpt.get_value({'M_nicholas': 'x'}), 29999/30000)
        self.assertAlmostEqual(cpt.get_value({'M_nicholas': 'X'}), 1/30000)

    def test_create_cpt_alexey(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_maternal_inheritance_cpt(alexey)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xx', 'M_alexey': 'x'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xx', 'M_alexey': 'X'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xX', 'M_alexey': 'x'}), 0.5)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xX', 'M_alexey': 'X'}), 0.5)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'XX', 'M_alexey': 'x'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'XX', 'M_alexey': 'X'}), 1.0)

    def test_create_cpt_anastasia(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_maternal_inheritance_cpt(anastasia)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xx', 'M_anastasia': 'x'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xx', 'M_anastasia': 'X'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xX', 'M_anastasia': 'x'}), 0.5)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'xX', 'M_anastasia': 'X'}), 0.5)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'XX', 'M_anastasia': 'x'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_alexandra': 'XX', 'M_anastasia': 'X'}), 1.0)


class TestNine(unittest.TestCase):

    def test_create_cpt_alexandra(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_paternal_inheritance_cpt(alexandra)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'x'}), 29999/30000)
        self.assertAlmostEqual(cpt.get_value({'P_alexandra': 'X'}), 1/30000)

    def test_create_cpt_anastasia(self):
        alexandra, nicholas, alexey, anastasia = romanoffs()
        cpt = create_paternal_inheritance_cpt(anastasia)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'xy', 'P_anastasia': 'x'}), 1.0)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'xy', 'P_anastasia': 'X'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'Xy', 'P_anastasia': 'x'}), 0.0)
        self.assertAlmostEqual(cpt.get_value({'G_nicholas': 'Xy', 'P_anastasia': 'X'}), 1.0)



class TestRomanoffs(unittest.TestCase):

    def test_romanoffs1(self):
        prob = compute_conditional_probability(create_family_bayes_net(romanoffs()),
                                               {'G_anastasia': 'xX'},
                                               {'H_alexey': '+'})
        self.assertAlmostEqual(prob, 0.5000166655555557)

    def test_romanoffs2(self):
        prob = compute_conditional_probability(create_family_bayes_net(romanoffs()),
                                               {'G_anastasia': 'xX'},
                                               {'H_alexey': '-'})
        self.assertAlmostEqual(prob, 4.9998888888888884e-05)

    def test_romanoffs3(self):
        prob = compute_conditional_probability(create_family_bayes_net(romanoffs()),
                                               {'H_alexandra': '-', 'H_nicholas': '-'},
                                               {'H_alexey': '+'})
        self.assertAlmostEqual(prob, 0.9999333344444444)


if __name__ == "__main__":
    unittest.main()   