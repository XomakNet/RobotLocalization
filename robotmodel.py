from collections import namedtuple
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from data_source import PhoneDataSource, RobotDataSource, CameraDataSource
from sensors_bag import SensorsBag

__author__ = 'Xomak'


class RobotModel:
    TACHO_COUNT_PER_ROTATION = 360
    RPM_TO_RADIANS = 0.104719755
    RPS_TO_RADIANS = 2 * np.pi
    WHEEL_RADIUS = 2.75
    WHEEL_DISTANCE = 14.5 / 2
    DISTANCE_TO_WALL = 13

    StepData = namedtuple('StepData', ('observations',
                                       'control',
                                       'observations_covariance',
                                       'process_covariance',
                                       'transition_function',
                                       'observations_function'))

    def __init__(self, robot_data: RobotDataSource, phone_data: PhoneDataSource, camera_data: CameraDataSource):

        self.camera_data = camera_data
        self.phone_data = phone_data
        self.robot_data = robot_data

    def get_transition_function_for_delta_time(self, delta_time: float):

        def transition_function(state: np.array, control: np.array) -> np.array:
            return self.transition_function(state, control, delta_time)

        return transition_function

    @property
    def main_timeline(self):
        return self.robot_data.timeline

    def get_step_data(self, timestamp: float, state: np.array, time_delta: float) -> StepData:
        observations, control = self.get_observations_and_control_for(timestamp)
        process_covariance = self.process_covariance
        transition_function = self.get_transition_function_for_delta_time(time_delta)
        observations_function = self.observations_function
        observations_covariance = self.observations_covariance(state, observations)
        result = RobotModel.StepData(observations, control, observations_covariance,
                                     process_covariance, transition_function, observations_function)
        return result

    def add_to_bag(self, step_data: 'RobotModel.StepData', sensor_bag: SensorsBag):
        sensor_bag.add_parameter('x', 'camera', step_data.observations[0])
        sensor_bag.add_parameter('y', 'camera', step_data.observations[1])
        sensor_bag.add_parameter('angle', 'compass', step_data.observations[2])
        sensor_bag.add_parameter('angle', 'gyroscope', step_data.observations[3])
        sensor_bag.add_parameter('sonar_distance', 'sonar', step_data.observations[4])
        distance = step_data.observations[4]
        angle = np.deg2rad(RobotModel.normalize_angle(step_data.observations[3]))
        sensor_bag.add_parameter('y', 'sonar', RobotModel.get_y_from_measured(distance, angle))
        return sensor_bag

    def get_observations_and_control_for(self, timestamp: float) -> Tuple[np.array, np.array]:

        camera_row = self.camera_data.find_nearest_to_timestamp(timestamp)
        camera_x = camera_row[1]
        camera_y = camera_row[2]
        phone_row = self.phone_data.find_nearest_to_timestamp(timestamp)
        phone_angle = RobotModel.normalize_angle(phone_row[1])
        robot_row = self.robot_data.find_nearest_to_timestamp(timestamp)
        robot_sonar = robot_row[1]
        robot_angle = RobotModel.normalize_angle(robot_row[2])
        left_motor = robot_row[3]
        right_motor = robot_row[4]

        observations = np.array([camera_x, camera_y, phone_angle, robot_angle, robot_sonar])
        control = np.array([left_motor, right_motor])

        return observations, control

    @staticmethod
    def normalize_angle(angle):
        return (angle + 360) % 360

    @staticmethod
    def transition_function(state: np.array, control: np.array, time_delta: float):
        x = state[0]
        y = state[1]
        theta = state[2]  # Theta in degrees
        theta_radians = np.radians(theta)

        left_motor_speed = control[0] / RobotModel.TACHO_COUNT_PER_ROTATION * RobotModel.RPS_TO_RADIANS
        right_motor_speed = control[1] / RobotModel.TACHO_COUNT_PER_ROTATION * RobotModel.RPS_TO_RADIANS

        velocity = RobotModel.WHEEL_RADIUS / 2 * (left_motor_speed + right_motor_speed)

        # Angular velocity in radians/s
        angular_velocity = RobotModel.WHEEL_RADIUS / (2 * RobotModel.WHEEL_DISTANCE) * (left_motor_speed - right_motor_speed)

        #print(velocity, angular_velocity)

        new_angle_radians = theta_radians + time_delta * angular_velocity

        x += time_delta * velocity * np.cos(new_angle_radians)
        y += time_delta * velocity * np.sin(new_angle_radians)

        # x += time_delta * velocity * np.cos(theta_radians)
        # y -= time_delta * velocity * np.sin(theta_radians)
        new_angle = np.degrees(new_angle_radians)

        new_angle = (new_angle + 360) % 360

        return np.array([x, y, new_angle])

    @staticmethod
    def get_y_from_measured(distance: float, angle_radians: float):
        return np.cos(angle_radians) * distance - RobotModel.DISTANCE_TO_WALL

    @staticmethod
    def get_measured_from_y_and_angle(y: float, angle_radians: float):
        return y / np.cos(angle_radians) + RobotModel.DISTANCE_TO_WALL

    @property
    def initial_covariance(self):
        return np.matrix("25 0 0; 0 25 0; 0 0 1")

    @property
    def initial_state(self):
        return np.array([0, 0, 0])

    @property
    def process_covariance(self):
        return np.matrix("25 0 0; 0 25 0; 0 0 4")

    def observations_covariance(self, state: np.array, observations: np.array):
        sonar_degree = 75

        covariance = np.matrix(np.zeros((5, 5)))
        covariance[0, 0] = 16
        covariance[1, 1] = 16
        covariance[2, 2] = 360*360
        covariance[3, 3] = 25

        if (state[2] < sonar_degree or state[2] > (360-sonar_degree)) and observations[4] < 100 and observations[4] > 13:
            covariance[4, 4] = 4
        else:
            covariance[4, 4] = 100000
        return covariance

    @property
    def observations_dimension(self):
        return 5

    def observations_function(self, state: np.array):
        # Observations: Camera X, Camera Y, Compass angle, Gyroscope angle, Sonar distance
        observed = np.ndarray((5,))
        observed[0] = state[0]  # camera x = state.x
        observed[1] = state[1]  # camera y = state.y
        observed[2] = state[2]  # Compass angle = theta
        observed[3] = state[2]  # Gyroscope angle = theta
        observed[4] = RobotModel.get_measured_from_y_and_angle(state[1], np.deg2rad(state[2]))  # Sonar measured, calc from y and theta
        return observed
