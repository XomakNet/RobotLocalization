from typing import Tuple

import numpy as np

from core.filters import DataFilter
from core.filters.particle_filter import ParticleFilter
from core.localization import Localization

__author__ = 'Xomak'


class PfLocalization(Localization):

    def init_filter(self) -> Tuple[DataFilter, np.array]:
        initial_state = self.robot_model.initial_state
        data_filter = ParticleFilter(initial_state.shape[0], self.robot_model.observations_dimension, 5000)
        data_filter.set_initial(initial_state, self.robot_model.initial_stds)
        return data_filter, initial_state

    @property
    def filtered_name(self):
        return "Particle filter"

    def get_next_state(self, data_filter: DataFilter, step_data):
        state = data_filter.predict_update(step_data.transition_function,
                                           step_data.observations_function,
                                           step_data.control,
                                           step_data.observations,
                                           step_data.control_stds,
                                           step_data.observations_covariance)
        return state

loc = PfLocalization()
loc.visualize()


