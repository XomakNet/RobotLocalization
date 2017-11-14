from typing import Callable
from typing import Tuple

from filterpy.kalman import ExtendedKalmanFilter, dot3
import numdifftools as nd
import numpy as np

__author__ = 'Xomak'


class ExtendedKalmanFilterLib(ExtendedKalmanFilter):
    def __init__(self, state_dimension: int, observations_dimension: int):
        super().__init__(state_dimension, observations_dimension)
        self.transition_function = None
        self.observations_function = None

    def set_initial(self, initial_state: np.array, initial_covariance: np.matrix):
        self.x = initial_state
        self.P = initial_covariance

    def predict(self, control=np.array([0])):
        self._x = self.transition_function(self._x, control)
        self._F = nd.Jacobian(self.transition_function)(self._x, control)
        self._P = dot3(self._F, self._P, self._F.T) + self._Q

    def predict_update(self,
                       transition_function: Callable[[np.array, np.array], np.array],
                       observations_function: Callable[[np.array], np.array],
                       control_input: np.array,
                       observed: np.array,
                       process_covariance: np.matrix,
                       observations_covariance: np.matrix) -> Tuple[np.array, np.matrix]:
        self.transition_function = transition_function
        self.observations_function = observations_function
        self._Q = process_covariance

        def find_jacobian(state, *args):
            return nd.Jacobian(self.observations_function)(state)

        self.predict(control_input)
        self.update(observed, find_jacobian, self.observations_function, observations_covariance)
        self._x = np.squeeze(np.asarray(self._x))
        self._P = np.matrix(self._P)

        return self._x, self._P