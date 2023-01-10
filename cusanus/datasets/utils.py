#!/usr/bin/env python3

import math
import numpy as np

# from https://stackoverflow.com/a/17637351
class RunningStats:

    def __init__(self, d:int):
        self.n = 0
        self.d = d
        self.old_m = np.zeros(d)
        self.new_m = np.zeros(d)
        self.old_s = np.zeros(d)
        self.new_s = np.zeros(d)

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros(self.d)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else np.zeros(self.d)

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())
