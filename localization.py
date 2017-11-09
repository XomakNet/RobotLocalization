import numpy as np

from data_source import PhoneDataSource, RobotDataSource, CameraDataSource
from kalman import ExtendedKalmanFilter
from robotmodel import RobotModel
import matplotlib.pyplot as plt

__author__ = 'Xomak'


class Localization:
    def __init__(self, filter_class):
        self.filter_class = filter_class
        self.filter = None
        data_phone = PhoneDataSource('datasets/1/data_phone_good_1.csv', 179, 457)
        data_camera = CameraDataSource('datasets/1/log_camera.csv')
        data_robot = RobotDataSource('datasets/1/log_robot.csv')

        print(data_robot.timeline[-1])

        self.robot_model = RobotModel(data_robot, data_phone, data_camera)

    def simulate(self):
        state = None
        model_state = None
        previous_timestamp = None
        x_p = []
        y_p = []

        x_c = []
        y_c = []

        x = []
        y = []

        a_p = []
        a_g = []
        a_c = []
        a = []

        sonar = []

        for timestamp in self.robot_model.main_timeline:
            if previous_timestamp is None:
                state = self.robot_model.initial_state
                model_state = self.robot_model.initial_state
                initial_covariance = self.robot_model.initial_covariance
                self.filter = self.filter_class(state, initial_covariance)
            else:
                time_delta = timestamp - previous_timestamp
                observations, control = self.robot_model.get_observations_and_control_for(timestamp)
                observations_covariance = self.robot_model.observations_covariance(state, observations)
                process_covariance = self.robot_model.process_covariance
                transition_function = self.robot_model.get_transition_function_for_delta_time(time_delta)
                observations_function = self.robot_model.observations_function
                try:
                    state, covariance = self.filter.get_new_state(transition_function, observations_function, control,
                                                                  observations,
                                                                  process_covariance, observations_covariance)

                    model_state = self.robot_model.transition_function(model_state, control, time_delta)
                    x_p.append(model_state[0])
                    y_p.append(model_state[1])
                    x_c.append(observations[0])
                    y_c.append(observations[1])
                    x.append(state[0])
                    y.append(state[1])

                    sonar.append(observations[4])

                    a_p.append(RobotModel.normalize_angle(model_state[2]))
                    a_c.append(observations[2])
                    a_g.append(RobotModel.normalize_angle(observations[3]))
                    a.append(state[2])
                except:
                    print("exception")

            previous_timestamp = timestamp

        plt.plot(x_p, y_p, 'ro')
        plt.plot(x_c, y_c, 'go')
        plt.plot(x, y, 'yo')
        # plt.plot(a_c)
        # plt.plot(a_g)
        # plt.plot(sonar)
        # plt.plot(a_p)
        # plt.plot(a)
        plt.show()


t = Localization(ExtendedKalmanFilter)
t.simulate()