import unittest

import numpy as np

from core.filters.kalman import ExtendedKalmanFilter

__author__ = 'Xomak'


class ExtendedKalmanFilterTest(unittest.TestCase):
    def test_simple_one_dimensional_case(self):
        initial_state = np.array([0], dtype='float64')
        covariance_estimate = np.matrix('0.5', dtype='float64')

        def transition_function(state: np.array, control: np.array):
            return np.array([state[0] + control[0]])

        def observations_function(state: np.array):
            return state

        filter = ExtendedKalmanFilter(1, 1)
        filter.set_initial(initial_state, np.matrix('0', dtype='float64'))

        control_input = np.array([1], dtype='float64')
        observed = np.array([1], dtype='float64')

        actual_state, actual_covariance_estimate = filter.predict_update(transition_function, observations_function,
                                                                         control_input, observed, covariance_estimate,
                                                                         covariance_estimate)

        self.assertAlmostEqual((actual_covariance_estimate - np.matrix([0.25]))[0, 0], 0)
        self.assertEqual(actual_state, np.array([1]))

    def test_simple_two_dimensional_case(self):
        initial_state = np.array([0, 0], dtype='float64')
        covariance_estimate = np.matrix('0.5, 0; 0, 0.5', dtype='float64')

        def transition_function(state: np.array, control: np.array):
            return np.array([state[0] + control[0], state[1] + control[1]])

        def observations_function(state: np.array):
            return state

        filter = ExtendedKalmanFilter(2, 2)
        filter.set_initial(initial_state, np.matrix('0, 0; 0, 0', dtype='float64'))

        control_input = np.array([1, 1], dtype='float64')
        observed = np.array([1, 1], dtype='float64')

        actual_state, actual_covariance_estimate = filter.predict_update(transition_function, observations_function,
                                                                         control_input, observed, covariance_estimate,
                                                                         covariance_estimate)

        required_covariance = np.matrix("0.25 0; 0 0.25")
        self.assertTrue(np.allclose(actual_covariance_estimate, required_covariance))

        required_state = np.array([1, 1])
        self.assertEqual(actual_state[0], required_state[0])
        self.assertEqual(actual_state[1], required_state[1])

    def test_from_paper(self):
        # http://mplab.ucsd.edu/tutorials/Kalman.pdf, appendix 9

        def transition_function(state: np.array, control: np.array):
            a = np.matrix("1 -0.5; 0.5 1")
            result = np.asarray(a.dot(state)).flatten()
            return result

        def observation_function(state: np.array):
            b = np.matrix("1 2")
            return np.asarray(b.dot(state)).flatten()

        process_covariance = np.matrix("1 0; 0 1")
        observation_covariance = np.matrix("1")

        initial_covariance = np.matrix("1 0; 0 1")
        initial_state = np.array([1, -1])
        observations = [-2, 4.5, 1.75, 7.625]
        expected_values = [
            [1.04081633, -1.41836735],
            [3.53106567, 0.25088478],
            [0.94444962, 0.66461669],
            [2.68852441, 2.25424821]
        ]

        filter = ExtendedKalmanFilter(2, 1)
        filter.set_initial(initial_state, initial_covariance)
        for i in range(0, 4):
            observed = np.array([observations[i]])
            state, covariance = filter.predict_update(transition_function, observation_function, np.array([0]),
                                                      observed, process_covariance, observation_covariance)

            required_state = np.array(expected_values[i])
            self.assertTrue(np.allclose(required_state, state))

            # self.assertTrue(False)
