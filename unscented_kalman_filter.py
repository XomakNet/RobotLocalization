from typing import List, Callable, Tuple

import numpy as np

__author__ = 'Xomak'


class UnscentedKalmanFilter:
    def __init__(self, state_dimension: int, observations_dimension: int):
        self.observations_dimension = observations_dimension
        self.state_predicted = np.zeros((state_dimension,))
        self.state_dimension = state_dimension
        self.covariance_predicted = np.matrix(np.zeros((state_dimension, state_dimension)))
        self._alpha = 1
        self._beta = 2
        self._kappa = 0
        self._calculate_coefficients()

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def kappa(self):
        return self._kappa

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self._calculate_coefficients()

    @beta.setter
    def beta(self, value):
        self._alpha = value
        self._calculate_coefficients()

    @kappa.setter
    def kappa(self, value):
        self._alpha = value
        self._calculate_coefficients()

    def _calculate_coefficients(self):
        self.c = np.power(self._alpha, 2) * (self.state_dimension + self._kappa)
        self.lambda_coeff = np.power(self._alpha, 2) * (self.state_dimension + self._kappa) - self.state_dimension

    def set_initial(self, initial_state: np.array, initial_covariance_estimate: np.matrix):
        if initial_state.shape != (self.state_dimension,):
            raise ValueError("Incorrect dimension for initial state")

        if initial_covariance_estimate.shape != (self.state_dimension, self.state_dimension):
            raise ValueError("Incorrect dimension for state covariance")

        self.state_predicted = initial_state
        self.covariance_predicted = initial_covariance_estimate

    def _get_sigma_points(self, mean: np.array, covariance: np.matrix) -> List[np.array]:
        print(covariance)
        m = np.linalg.cholesky(self.c * covariance)
        points = [mean]

        for i in range(0, self.state_dimension):
            m_i = np.asarray(m[:, i]).reshape((self.state_dimension,))
            points.append(mean + m_i)
            points.append(mean - m_i)

        return points

    def _get_combined_point(self, points):
        w_s_0 = self.lambda_coeff / (self.lambda_coeff + self.state_dimension)
        w_s_i = 1 / (2 * (self.state_dimension + self.lambda_coeff))
        state = None
        for idx, point in enumerate(points):
            w_s = w_s_0 if idx == 0 else w_s_i
            weighted_point = w_s * point
            state = weighted_point if state is None else state + weighted_point

        return state

    def _get_covariance_weights(self):
        w_c_0 = self.lambda_coeff / (self.lambda_coeff + self.state_dimension) + (
            1 - np.power(self._alpha, 2) + self._beta)
        w_c_i = 1 / (2 * (self.state_dimension + self.lambda_coeff))
        return w_c_0, w_c_i

    def _get_cross_covariance(self, points_1, combined_point_1, points_2, combined_point_2):
        covariance = None
        first_dimension = combined_point_1.shape[0]
        second_dimension = combined_point_2.shape[0]
        for idx, points in enumerate(zip(points_1, points_2)):
            point_1, point_2 = points
            step_covariance = np.asmatrix((point_1 - combined_point_1).reshape((first_dimension, 1)) \
                                          .dot((point_2 - combined_point_2).reshape(second_dimension, 1).T))
            covariance = step_covariance if covariance is None else covariance + step_covariance

        return covariance

    def _get_self_covariance(self, points, combined_point):
        w_c_0, w_c_i = self._get_covariance_weights()
        covariance = None
        for idx, point in enumerate(points):
            w_c = w_c_0 if idx == 0 else w_c_i
            difference = (point - combined_point).reshape((point.shape[0], 1))
            weighted_covariance = w_c * (difference.dot(difference.T))
            covariance = weighted_covariance if covariance is None else covariance + weighted_covariance

        return np.asmatrix(covariance)

    def _fix_covariance(self, covariance):
        covariance = 0.5 * covariance + 0.5 * covariance
        covariance += np.identity(covariance.shape[0]) * 1e-1
        return covariance

    def _predict(self, transition_function: Callable[[np.array, np.array], np.array], control_input: np.array,
                 process_covariance: np.matrix) -> None:
        sigma_points = self._get_sigma_points(self.state_predicted, self.covariance_predicted)
        states_predicted = [transition_function(point, control_input) for point in sigma_points]
        self.state_predicted = self._get_combined_point(states_predicted)
        self.covariance_predicted = self._get_self_covariance(sigma_points, self.state_predicted) + process_covariance
        self.covariance_predicted = self._fix_covariance(self.covariance_predicted)

    def _update(self, observations_function: Callable[[np.array], np.array], observed: np.array,
                observations_covariance: np.matrix) -> None:
        sigma_points = self._get_sigma_points(self.state_predicted, self.covariance_predicted)
        measurements_predicted = [observations_function(point) for point in sigma_points]
        measurement_predicted = self._get_combined_point(measurements_predicted)
        self_covariance = self._get_self_covariance(measurements_predicted, measurement_predicted) \
                          + observations_covariance

        self_covariance = self._fix_covariance(self_covariance)
        #
        cross_covariance = 1/(2*self.c) * self._get_cross_covariance(sigma_points, self.state_predicted,
                                                      measurements_predicted, measurement_predicted)
        kalman_gain = cross_covariance.dot(self_covariance.I)
        self.state_predicted += np.asarray(kalman_gain.dot(observed - measurement_predicted)) \
            .reshape(self.state_dimension, )
        self.covariance_predicted -= kalman_gain.dot(self_covariance).dot(kalman_gain.T)
        self.covariance_predicted = self._fix_covariance(self.covariance_predicted)

    def predict_update(self,
                       transition_function: Callable[[np.array, np.array], np.array],
                       observations_function: Callable[[np.array], np.array],
                       control_input: np.array,
                       observed: np.array,
                       process_covariance: np.matrix,
                       observations_covariance: np.matrix) -> Tuple[np.array, np.matrix]:
        self._predict(transition_function, control_input, process_covariance)
        self._update(observations_function, observed, observations_covariance)
        return self.state_predicted, self.covariance_predicted
