from typing import Any

__author__ = 'Xomak'


class SensorsBag:

    def __init__(self):
        self.data = {}

    def add_parameter(self, sensor_type: str, sensor_name: str, value: Any):

        if sensor_type not in self.data:
            self.data[sensor_type] = {}

        if sensor_name not in self.data[sensor_type]:
            self.data[sensor_type][sensor_name] = []

        self.data[sensor_type][sensor_name].append(value)

    def get_values(self, sensor_type: str, sensor_name: str):
        return self.data[sensor_type][sensor_name]
