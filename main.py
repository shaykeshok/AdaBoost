import Line
from AdaBoost import AdaBoost
from DataSet import DataSet

# r-> number of iteration for the adaboost function (1...8)
# iterations->number of iterations of the algorithem(we were asked for 100)
# returns string to save in a file

r = 9  # equal to nine because range(1..9) return 1,2,...,8 without 9
iterations = 100
# HC_Body_Temperature dataSet
temperature_file = DataSet('./data_files/HC_Body_Temperature.txt', 'HC_Body_Temperature', sep=r"\s+")
adaboost = AdaBoost(temperature_file.df, Line, r, iterations)
adaboost.active()

# Iris dataSet
iris_file = DataSet('./data_files/iris.data', 'iris')

# iris_file.print_data()
# temperature_file.print_data()
