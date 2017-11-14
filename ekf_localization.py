from typing import Tuple

import numpy as np

from core.filters import DataFilter
from core.filters.kalman import ExtendedKalmanFilter
from core.localization import Localization

__author__ = 'Xomak'


class EkfLocalization(Localization):
    def init_filter(self) -> Tuple[DataFilter, np.array]:
        initial_state = self.robot_model.initial_state
        data_filter = ExtendedKalmanFilter(initial_state.shape[0], self.robot_model.observations_dimension)
        data_filter.set_initial(initial_state, self.robot_model.initial_covariance)
        return data_filter, initial_state

    @property
    def filtered_name(self):
        return "EK filter"

    def get_next_state(self, data_filter: DataFilter, step_data):
        state, covariance = data_filter.predict_update(step_data.transition_function,
                                                       step_data.observations_function,
                                                       step_data.control,
                                                       step_data.observations,
                                                       step_data.process_covariance,
                                                       step_data.observations_covariance)
        return state

loc = EkfLocalization()
loc.visualize()
