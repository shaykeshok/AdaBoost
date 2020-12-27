import Line
from AdaBoost import AdaBoost
from DataSet import DataSet


# number of iteration for the adaboost function (1...8)
# equal to nine because range(1..9) return 1,2,...,8 without 9
k = 9
# number of iterations of the algorithm
iterations = 20


# HC_Body_Temperature dataSet
print("\n**************  HC_Body_Temperature dataSet  **************")
temperature_file = DataSet('./data_files/HC_Body_Temperature.txt', 'HC_Body_Temperature', sep=r"\s+")
adaboost = AdaBoost(temperature_file.df, k, iterations)
adaboost.active()

# Iris dataSet
print("\n**************  iris dataSet  **************")
iris_file = DataSet('./data_files/iris.data', 'iris')
adaboost = AdaBoost(iris_file.df, k, iterations)
adaboost.active()


# iris_file.print_data()
# temperature_file.print_data()
