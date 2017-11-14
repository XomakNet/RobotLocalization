from typing import Tuple, Callable

import numpy as np

from core.data_source import PhoneDataSource, RobotDataSource, CameraDataSource
from core.filters import DataFilter
from core.sensors_bag import SensorsBag
from core.visualizer import Visualizer
from robotmodel import RobotModel

__author__ = 'Xomak'


class Localization:
    def __init__(self):
        self.robot_model = None
        self.init_model()

    @property
    def filtered_name(self):
        raise ValueError()

    def init_model(self):
        data_phone = PhoneDataSource('datasets/1/data_phone_good_1.csv', 179, 457)
        data_camera = CameraDataSource('datasets/1/log_camera.csv')
        data_robot = RobotDataSource('datasets/1/log_robot.csv')
        self.robot_model = RobotModel(data_robot, data_camera)

    def init_filter(self) -> Tuple[DataFilter, np.array]:
        raise NotImplemented()

    def get_next_state(self, data_filter: DataFilter, step_data):
        raise NotImplemented()

    def simulate(self):
        state = None
        model_state = None
        previous_timestamp = None

        sensors_bag = SensorsBag()
        data_filter = None

        for timestamp in self.robot_model.main_timeline:
            if previous_timestamp is None:
                data_filter, state = self.init_filter()
                model_state = state
            else:
                time_delta = timestamp - previous_timestamp
                step_data = self.robot_model.get_step_data(timestamp, state, time_delta)

                predicted_state = self.robot_model.transition_function(state, step_data.control, time_delta)
                state = self.get_next_state(data_filter, step_data)

                model_state = self.robot_model.transition_function(model_state, step_data.control, time_delta)
                sensors_bag = self.robot_model.add_to_bag(step_data, sensors_bag)

                sensors_bag.add_parameter('x', 'predicted', predicted_state[0])
                sensors_bag.add_parameter('y', 'predicted', predicted_state[1])
                sensors_bag.add_parameter('angle', 'predicted', predicted_state[2])

                sensors_bag.add_parameter('x', 'model', model_state[0])
                sensors_bag.add_parameter('y', 'model', model_state[1])
                sensors_bag.add_parameter('angle', 'model', model_state[2])

                sensors_bag.add_parameter('x', self.filtered_name, state[0])
                sensors_bag.add_parameter('y', self.filtered_name, state[1])
                sensors_bag.add_parameter('angle', self.filtered_name, state[2])

            previous_timestamp = timestamp

        return sensors_bag

    def visualize(self):

        vis = Visualizer(self.simulate())

        def show_window(func: Callable[[None], None]):
            vis.figure()
            func()
            vis.show()

        show_window(lambda: vis.plot_xy(['model', 'camera', self.filtered_name]))
        show_window(lambda: vis.plot(['x'], ['camera', 'model', self.filtered_name]))
        show_window(lambda: vis.plot(['y'], ['camera', 'model', 'sonar', self.filtered_name]))
        show_window(lambda: vis.plot(['angle'], ['model', 'gyroscope', self.filtered_name]))
