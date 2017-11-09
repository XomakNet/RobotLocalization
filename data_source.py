import numpy as np

__author__ = 'Xomak'


class DataSource:

    @staticmethod
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        if(np.abs(array[idx] - value)) > 1:
            raise ValueError("Not found: {}".format(value))
        return idx

    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter=';', skiprows=1)

    def find_nearest_to_timestamp(self, timestamp: float):
        idx = DataSource.find_nearest(self.data[:, 0], timestamp)
        return self.data[idx, :]

    @property
    def timeline(self):
        return self.data[:, 0]

    def __iter__(self):
        return self.data.__iter__()


class CameraDataSource(DataSource):
    pass


class RobotDataSource(DataSource):
    pass


class PhoneDataSource(DataSource):

    def __init__(self, filename, first_row_to_consider: int, last_stable_row: int):
        super().__init__(filename)
        self.init_value = np.mean(self.data[first_row_to_consider:last_stable_row, 1])
        self.current_iterator = None

    def find_nearest_to_timestamp(self, timestamp: float):
        idx = DataSource.find_nearest(self.data[:, 0], timestamp)
        # print(idx)
        # print(timestamp, self.data[idx, 0])
        return self.proceed_row(self.data[idx, :])

    def proceed_row(self, row: np.ndarray):
        row[1] = row[1] - self.init_value
        return row

    def __iter__(self):
        self.current_iterator = self.data.__iter__()
        return self

    def __next__(self):
        row = self.current_iterator.__next__()
        return self.proceed_row(row)
