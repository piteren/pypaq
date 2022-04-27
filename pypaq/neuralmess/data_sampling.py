from collections import deque
import random
from typing import Iterable, Sized


class DataSampler:

    def __init__(
            self,
            data: Sized and Iterable,
            seed = 123):
        self.__seed_counter = seed
        self.__data = data
        self.__data_shuffled = self.__get_data_shuffled()

    # prepares new deque of shuffled data
    def __get_data_shuffled(self) -> deque:
        random.seed(self.__seed_counter)
        self.__seed_counter += 1
        indexes = list(range(len(self.__data)))
        random.shuffle(indexes)
        # print(indexes)
        data_shuffled = [self.__data[i] for i in indexes]
        # print(data_shuffled)
        data_shuffled = deque(data_shuffled)
        return data_shuffled

    def get_sample(self):
        if not self.__data_shuffled:
            self.__data_shuffled = self.__get_data_shuffled()
        return self.__data_shuffled.popleft()

    def epoch_finished(self) -> bool:
        return not self.__data_shuffled


if __name__ == '__main__':

    data = [1,2,3,4,5,6,7,8]
    ds = DataSampler(data)
    for _ in range(20):
        print(ds.get_sample())
        if ds.epoch_finished(): print('epoch finished')