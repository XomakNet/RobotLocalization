from collections import namedtuple
from typing import Tuple

import numpy as np

from data_source import PhoneDataSource, RobotDataSource, CameraDataSource
from sensors_bag import SensorsBag

__author__ = 'Xomak'


class RobotModel:
    TACHO_COUNT_PER_ROTATION = 360
    RPM_TO_RADIANS = 0.104719755
    RPS_TO_RADIANS = 2 * np.pi
    WHEEL_RADIUS = 2.75
    WHEEL_DISTANCE = 14.5 / 2
    SONAR_ENABLED_ANGLE = 45

    StepData = namedtuple('StepData', ('observations',
                                       'control',
                                       'observations_covariance',
                                       'process_covariance',
                                       'transition_function',
                                       'observations_function'))

    ProcessDispersions = namedtuple('ProcessDispersions', ('x', 'y', 'theta'))

    ObservationDispersions = namedtuple('ObservationDispersionsWithoutCompass',
                                        ('camera_x', 'camera_y', 'angle_gyro',
                                         'sonar', 'sonar_inf'))

    def __init__(self, robot_data: RobotDataSource, camera_data: CameraDataSource):

        self.camera_data = camera_data
        self.robot_data = robot_data
        self.distance_to_wall = 13
        self.initial_dispersions = self.ProcessDispersions(25, 25, 4)
        self.process_dispersions = self.ProcessDispersions(225, 225, 225)
        self.observation_dispersions = self.ObservationDispersions(100, 100, 4, 4, 100000)

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
        observations_covariance = self.observations_covariance(observations)
        result = RobotModel.StepData(observations, control, observations_covariance,
                                     process_covariance, transition_function, observations_function)
        return result

    def add_to_bag(self, step_data: 'RobotModel.StepData', sensor_bag: SensorsBag):
        # Without compass

        sensor_bag.add_parameter('x', 'camera', step_data.observations[0])
        sensor_bag.add_parameter('y', 'camera', step_data.observations[1])
        sensor_bag.add_parameter('angle', 'gyroscope', step_data.observations[2])
        sensor_bag.add_parameter('sonar_distance', 'sonar', step_data.observations[3])
        distance = step_data.observations[3]
        angle = np.deg2rad(self.normalize_angle(step_data.observations[2]))
        sensor_bag.add_parameter('y', 'sonar', self.get_y_from_measured(distance, angle))
        return sensor_bag

    def get_observations_and_control_for(self, timestamp: float) -> Tuple[np.array, np.array]:
        # Without compass

        camera_row = self.camera_data.find_nearest_to_timestamp(timestamp)
        camera_x = camera_row[1]
        camera_y = camera_row[2]
        robot_row = self.robot_data.find_nearest_to_timestamp(timestamp)
        robot_sonar = robot_row[1]
        robot_angle = self.normalize_angle(robot_row[2])
        left_motor = robot_row[3]
        right_motor = robot_row[4]

        observations = np.array([camera_x, camera_y, robot_angle, robot_sonar])
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
        angular_velocity = RobotModel.WHEEL_RADIUS / (2 * RobotModel.WHEEL_DISTANCE) * (
            left_motor_speed - right_motor_speed)

        new_angle_radians = theta_radians + time_delta * angular_velocity

        x += time_delta * velocity * np.cos(new_angle_radians)
        y += time_delta * velocity * np.sin(new_angle_radians)

        new_angle = np.degrees(new_angle_radians)

        new_angle = RobotModel.normalize_angle(new_angle)

        return np.array([x, y, new_angle])

    def get_y_from_measured(self, distance: float, angle_radians: float):
        return np.cos(angle_radians) * distance - self.distance_to_wall

    def get_measured_from_y_and_angle(self, y: float, angle_radians: float):
        return y / np.cos(angle_radians) + self.distance_to_wall

    @property
    def initial_covariance(self):
        matrix = np.asmatrix(np.zeros((3, 3)))
        matrix[0, 0] = self.initial_dispersions.x
        matrix[1, 1] = self.initial_dispersions.y
        matrix[2, 2] = self.initial_dispersions.theta
        return matrix

    @property
    def initial_state(self):
        return np.array([0, 0, 0])

    @property
    def process_covariance(self):
        matrix = np.asmatrix(np.zeros((3, 3)))
        matrix[0, 0] = self.process_dispersions.x
        matrix[1, 1] = self.process_dispersions.y
        matrix[2, 2] = self.process_dispersions.theta
        return matrix

    @property
    def observations_dimension(self):
        return 4

    def observations_function(self, state: np.array):
        # WITHOUT
        # Observations: Camera X, Camera Y, Gyroscope angle, Sonar distance
        observed = np.ndarray((4,))
        observed[0] = state[0]  # camera x = state.x
        observed[1] = state[1]  # camera y = state.y
        observed[2] = state[2]  # Gyroscope angle = theta
        observed[3] = self.get_measured_from_y_and_angle(state[1],
                                                         np.deg2rad(state[2]))  # Sonar measured, calc from y and theta
        return observed

    def observations_covariance(self, observations: np.array):
        # Without compass
        covariance = np.matrix(np.zeros((4, 4)))
        covariance[0, 0] = self.observation_dispersions.camera_x
        covariance[1, 1] = self.observation_dispersions.camera_y
        covariance[2, 2] = self.observation_dispersions.angle_gyro

        sonar_distance = observations[3]
        angle = observations[2]

        if (angle < self.SONAR_ENABLED_ANGLE or angle > (360 - self.SONAR_ENABLED_ANGLE)) \
                and 100 > sonar_distance > self.distance_to_wall:
            covariance[3, 3] = self.observation_dispersions.sonar
        else:
            covariance[3, 3] = self.observation_dispersions.sonar_inf

        return covariance


class RobotModelWithCompass(RobotModel):
    ObservationDispersionsWithCompass = namedtuple('ObservationDispersions',
                                                   ('camera_x', 'camera_y', 'angle_gyro',
                                                    'sonar', 'sonar_inf', 'angle_compass'))

    def __init__(self, robot_data: RobotDataSource, camera_data: CameraDataSource, phone_data: PhoneDataSource):
        super().__init__(robot_data, camera_data)
        self.observation_dispersions = self.ObservationDispersionsWithCompass(100, 100, 4, 4, 100000, 360*360)
        self.phone_data = phone_data

    @property
    def observations_dimension(self):
        return 5

    def add_to_bag(self, step_data: 'RobotModel.StepData', sensor_bag: SensorsBag):
        sensor_bag.add_parameter('x', 'camera', step_data.observations[0])
        sensor_bag.add_parameter('y', 'camera', step_data.observations[1])
        sensor_bag.add_parameter('angle', 'compass', step_data.observations[2])
        sensor_bag.add_parameter('angle', 'gyroscope', step_data.observations[3])
        sensor_bag.add_parameter('sonar_distance', 'sonar', step_data.observations[4])
        distance = step_data.observations[4]
        angle = np.deg2rad(RobotModel.normalize_angle(step_data.observations[3]))
        sensor_bag.add_parameter('y', 'sonar', self.get_y_from_measured(distance, angle))
        return sensor_bag

    def get_observations_and_control_for(self, timestamp: float) -> Tuple[np.array, np.array]:

        camera_row = self.camera_data.find_nearest_to_timestamp(timestamp)
        camera_x = camera_row[1]
        camera_y = camera_row[2]
        phone_row = self.phone_data.find_nearest_to_timestamp(timestamp)
        phone_angle = self.normalize_angle(phone_row[1])
        robot_row = self.robot_data.find_nearest_to_timestamp(timestamp)
        robot_sonar = robot_row[1]
        robot_angle = self.normalize_angle(robot_row[2])
        left_motor = robot_row[3]
        right_motor = robot_row[4]

        observations = np.array([camera_x, camera_y, phone_angle, robot_angle, robot_sonar])
        control = np.array([left_motor, right_motor])

        return observations, control

    def observations_covariance(self, observations: np.array):

        covariance = np.matrix(np.zeros((5, 5)))
        covariance[0, 0] = self.observation_dispersions.camera_x
        covariance[1, 1] = self.observation_dispersions.camera_y
        covariance[2, 2] = self.observation_dispersions.angle_compass
        covariance[3, 3] = self.observation_dispersions.angle_gyro

        sonar_distance = observations[4]
        angle = observations[3]

        if (angle < self.SONAR_ENABLED_ANGLE or angle > (360 - self.SONAR_ENABLED_ANGLE)) \
                and 100 > sonar_distance > self.distance_to_wall:
            covariance[4, 4] = self.observation_dispersions.sonar
        else:
            covariance[4, 4] = self.observation_dispersions.sonar_inf

        return covariance

    def observations_function(self, state: np.array):
        # Observations: Camera X, Camera Y, Compass angle, Gyroscope angle, Sonar distance
        observed = np.ndarray((5,))
        observed[0] = state[0]  # camera x = state.x
        observed[1] = state[1]  # camera y = state.y
        observed[2] = state[2]  # Compass angle = theta
        observed[3] = state[2]  # Gyroscope angle = theta
        observed[4] = self.get_measured_from_y_and_angle(state[1],
                                                         np.deg2rad(state[2]))  # Sonar measured, calc from y and theta
        return observed
