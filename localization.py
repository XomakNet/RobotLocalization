
from data_source import PhoneDataSource, RobotDataSource, CameraDataSource
from kalman import ExtendedKalmanFilter
from robotmodel import RobotModel

from sensors_bag import SensorsBag
from visualizer import Visualizer

__author__ = 'Xomak'


class Localization:
    def __init__(self, filter_class):
        self.filter_class = filter_class
        self.filter = None
        data_phone = PhoneDataSource('datasets/1/data_phone_good_1.csv', 179, 457)
        data_camera = CameraDataSource('datasets/1/log_camera.csv')
        data_robot = RobotDataSource('datasets/1/log_robot.csv')

        print(data_robot.timeline[-1])

        self.robot_model = RobotModel(data_robot, data_camera)

    def simulate(self):
        state = None
        model_state = None
        previous_timestamp = None

        sensors_bag = SensorsBag()

        for timestamp in self.robot_model.main_timeline:
            if previous_timestamp is None:
                state = self.robot_model.initial_state
                model_state = self.robot_model.initial_state
                initial_covariance = self.robot_model.initial_covariance
                self.filter = self.filter_class(state.shape[0], self.robot_model.observations_dimension)
                self.filter.set_initial(state, initial_covariance)
            else:
                time_delta = timestamp - previous_timestamp
                step_data = self.robot_model.get_step_data(timestamp, state, time_delta)

                predicted_state = self.robot_model.transition_function(state, step_data.control, time_delta)
                state, covariance = self.filter.predict_update(step_data.transition_function,
                                                               step_data.observations_function,
                                                               step_data.control,
                                                               step_data.observations,
                                                               step_data.process_covariance,
                                                               step_data.observations_covariance)

                model_state = self.robot_model.transition_function(model_state, step_data.control, time_delta)
                sensors_bag = self.robot_model.add_to_bag(step_data, sensors_bag)

                sensors_bag.add_parameter('x', 'predicted', predicted_state[0])
                sensors_bag.add_parameter('y', 'predicted', predicted_state[1])
                sensors_bag.add_parameter('angle', 'predicted', predicted_state[2])

                sensors_bag.add_parameter('x', 'model', model_state[0])
                sensors_bag.add_parameter('y', 'model', model_state[1])
                sensors_bag.add_parameter('angle', 'model', model_state[2])

                sensors_bag.add_parameter('x', 'kalman', state[0])
                sensors_bag.add_parameter('y', 'kalman', state[1])
                sensors_bag.add_parameter('angle', 'kalman', state[2])


            previous_timestamp = timestamp

        return sensors_bag


t = Localization(ExtendedKalmanFilter)
data = t.simulate()
vis = Visualizer(data)
#vis.plot_xy(['model', 'camera', 'kalman'])
vis.plot(['y'], ['camera', 'model', 'kalman', 'sonar'])
# vis.plot(['angle'], ['gyroscope', 'model', 'compass', 'kalman'])
vis.show()
