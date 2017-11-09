from typing import Callable, Tuple

import numdifftools as nd
import numpy as np

__author__ = 'Xomak'


class ExtendedKalmanFilter:
    def __init__(self, state_dimension: int, observations_dimension: int, ):
        self.observations_dimension = observations_dimension
        self.state_predicted = np.zeros((state_dimension,))
        self.state_dimension = state_dimension
        self.covariance_predicted = np.matrix(np.zeros((state_dimension, state_dimension)))

    def set_initial(self, initial_state: np.array, initial_covariance_estimate: np.matrix):
        if initial_state.shape != (self.state_dimension, ):
            raise ValueError("Incorrect dimension for initial state")

        if initial_covariance_estimate.shape != (self.state_dimension, self.state_dimension):
            raise ValueError("Incorrect dimension for state covariance")

        self.state_predicted = initial_state
        self.covariance_predicted = initial_covariance_estimate

    def _predict(self, transition_function: Callable[[np.array, np.array], np.array], control_input: np.array,
                 process_covariance: np.matrix) -> None:

        self.state_predicted = transition_function(self.state_predicted, control_input)
        transition_matrix = nd.Jacobian(transition_function)(self.state_predicted, control_input)
        self.covariance_predicted = transition_matrix \
                                        .dot(self.covariance_predicted) \
                                        .dot(transition_matrix.T) \
                                    + process_covariance

    def _update(self, observations_function: Callable[[np.array], np.array], observed: np.array,
                observations_covariance: np.matrix) -> None:
        measurement_residual = observed - observations_function(self.state_predicted)
        observations_matrix = nd.Jacobian(observations_function)(self.state_predicted)

        residual_covariance = observations_matrix \
                                  .dot(self.covariance_predicted) \
                                  .dot(observations_matrix.T) \
                              + observations_covariance
        kalman_gain = self.covariance_predicted.dot(observations_matrix.T).dot(residual_covariance.I)
        self.state_predicted += np.squeeze(np.asarray(kalman_gain.dot(measurement_residual)))
        self.covariance_predicted = (np.identity(kalman_gain.shape[0]) - kalman_gain.dot(observations_matrix)).dot(self.covariance_predicted)

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
