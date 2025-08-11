import numpy as np
import unittest
from gradescope_utils.autograder_utils.decorators import weight

# Handle both VS Code (relative import) and autograder (absolute import) contexts
try:
    from .assignment import generate_sample  # VS Code context
except ImportError:
    from assignment import generate_sample  # Autograder context

class TestDiceSample(unittest.TestCase):

    @weight(5)
    def test_1_dimensions(self):
        '''Testing that the dimensions of the sample are correct.'''
        actual = generate_sample((0.5, 0.5), 
                                 ((1/4, 1/4, 1/4, 1/4), 
                                  (1/5, 1/5, 1/5, 1/5, 1/5)),
                                 7, 10)
        self.assertEqual(np.shape(actual), (7, 10))

    @weight(5)
    def test_2_sampled_value_range(self):
        """Testing that the sampled values lie in the correct range and that 
        0 probabilities are handled correctly."""
        actual = generate_sample((0., 1.),
                                 ((1/3, 1/3, 1/3, 0.),
                                  (1/5, 1/5, 1/5, 2/5)), 
                                  20, 20)
        min_val = min([min(x) for x in actual])
        max_val = max([max(x) for x in actual])
        self.assertEqual(min_val, 0)
        self.assertEqual(max_val, 3)

    @weight(5)
    def test_3_sampled_value_range(self):
        """Testing that the sampled values lie in the correct range and that 
        0 probabilities are handled correctly."""
        actual = generate_sample((1., 0.), 
                                 ((1/3, 1/3, 1/3), (1/5, 1/5, 1/5, 1/5, 1/5)), 
                                 7, 10)
        min_val = min((min(x) for x in actual))
        max_val = max((max(x) for x in actual))
        self.assertGreaterEqual(min_val, 0)
        self.assertLessEqual(max_val, 2)

    @weight(5)
    def test_4_probabilities_in_right_ballpark(self):
        sample = generate_sample((0., 1.),
                                 ((1/3, 1/3, 1/3, 0.),
                                  (0., 1/5, 2/5, 2/5)), 
                                  50, 20)

        count_value_1 = np.count_nonzero(sample == 1)
        count_value_2 = np.count_nonzero(sample == 2)

        total_count = len(sample) * len(sample[0])

        # Proportions in the sample
        proportion_value_1 = count_value_1 / total_count
        proportion_value_2 = count_value_2 / total_count

        # Expected proportions based on die priors
        expected_proportion_value_1 = 1/5
        expected_proportion_value_2 = 2/5

        # Allowable error (for example, 5%)
        error_margin = 0.05

        # Check if the proportions are in the right ballpark
        self.assertTrue(abs(proportion_value_1 -
                            expected_proportion_value_1) < error_margin)
        self.assertTrue(abs(proportion_value_2 -
                            expected_proportion_value_2) < error_margin)
