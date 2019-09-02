# -*- coding: utf-8 -*
import numpy
from flyai.processor.base import Base


def Normalize(data):
    m = numpy.mean(data)
    mx = max(data)
    mn = min(data)
    return [(float(i) - m) / (mx - mn) for i in data]


class Processor(Base):

    def input_x(self, sepal_length, sepal_width, petal_length, petal_width):
        x_data = numpy.zeros(4)  ## 输入维度为13
        x_data[:] = sepal_length, sepal_width, petal_length, petal_width
        x_data = Normalize(x_data)

        return x_data

    def input_y(self, variety):
        #0,1,2
        if "Setosa" in variety:
            label = 0
        elif "Versicolor" in variety:
            label = 1
        elif "Virginica" in variety:
            label = 2
        
        return label

    def output_y(self, data):
        categorys = ["Setosa", "Versicolor", "Virginica"]
        return categorys[int(numpy.argmax(data))]