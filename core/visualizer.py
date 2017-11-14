from typing import List

import matplotlib.pyplot as plt

from core.sensors_bag import SensorsBag

__author__ = 'Xomak'


class Visualizer:

    def __init__(self, sensors_bag: SensorsBag):
        self.sensors_bag = sensors_bag
        self.legend_handles = []
        self.figure_id = 0

    def figure(self):
        self.figure_id += 1
        plt.figure(self.figure_id)

    def plot_xy(self, sensor_names: List[str]):
        for sensor_name in sensor_names:
            x = self.sensors_bag.get_values('x', sensor_name)
            y = self.sensors_bag.get_values('y', sensor_name)
            legend_handle, = plt.plot(x, y, 'o', label=sensor_name)
            self.legend_handles.append(legend_handle)

    def plot(self, sensor_types: [str], sensor_names: List[str]):
        for sensor_name in sensor_names:
            for sensor_type in sensor_types:
                data = self.sensors_bag.get_values(sensor_type, sensor_name)
                legend_handle, = plt.plot(data, label='{}: {}'.format(sensor_type, sensor_name))
                self.legend_handles.append(legend_handle)

    def show(self):
        plt.legend(handles=self.legend_handles)
        self.legend_handles = []
        plt.show()
