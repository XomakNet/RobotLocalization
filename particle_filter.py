import numpy as np
from typing import Callable, Tuple

import scipy
import scipy.stats

from filterpy.monte_carlo import stratified_resample

__author__ = 'Xomak'


class ParticleFilter:

    def __init__(self, state_dimension: int, observations_dimension: int, particles_number: int = 2000):
        self.observations_dimension = observations_dimension
        self.state_dimension = state_dimension
        self.particles = None
        self.weights = None
        self.particles_number = particles_number
        self.state_predicted = None

    def set_initial(self, initial_state: np.array, initial_covariance: np.matrix):
        self.state_predicted = initial_state
        stds = np.array([initial_covariance[0, 0], initial_covariance[1, 1], initial_covariance[2, 2]])
        self.init_particles(initial_state, np.array(stds), self.particles_number)

    def init_particles(self, initial_state, initial_stds, particles_number):
        self.particles = [initial_state + np.random.randn(initial_state.shape[0]) * initial_stds
                          for i in range(0, particles_number)]
        self.weights = np.ones((particles_number, ))
        self.weights.fill(1/particles_number)

    def _predict(self, transition_function: Callable[[np.array, np.array], np.array], control_input: np.array,
                 control_stds: np.array) -> None:

        def get_control_input():
            return control_input + np.random.randn(control_input.shape[0]) * control_stds

        self.particles = [transition_function(particle, get_control_input()) for particle in self.particles]

    def _update(self, observations_function: Callable[[np.array], np.array], observed: np.array,
                observations_covariance: np.matrix):

        for idx, particle in enumerate(self.particles):
            measurement = observations_function(particle)
            distribution = scipy.stats.multivariate_normal(measurement, observations_covariance)
            self.weights[idx] = distribution.pdf(observed)

        self.weights += 1.e-300
        self.weights /= sum(self.weights)

    def _calculate_predicted(self):
        self.state_predicted = np.zeros((self.state_dimension,))
        for idx, particle in enumerate(self.particles):
            self.state_predicted += self.weights[idx] * particle

        return self.state_predicted

    def _resampling_required(self):
        return 1. / np.sum(np.square(self.weights)) < len(self.particles)/2

    def _resample(self):
        indexes = stratified_resample(self.weights)
        self.particles = [self.particles[idx] for idx in indexes]
        self.weights[:] = self.weights[indexes]
        self.weights /= sum(self.weights)

    def predict_update(self,
                       transition_function: Callable[[np.array, np.array], np.array],
                       observations_function: Callable[[np.array], np.array],
                       control_input: np.array,
                       observed: np.array,
                       control_stds: np.array,
                       observations_stds: np.array) -> Tuple[np.array, np.matrix]:

        self._predict(transition_function, control_input, control_stds)
        self._update(observations_function, observed, observations_stds)

        if self._resampling_required():
            self._resample()

        self._calculate_predicted()

        return self.state_predicted
