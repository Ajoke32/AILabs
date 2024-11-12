from sklearn.datasets import load_iris
iris_dataset = load_iris()
#print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print(F"Full description: \n{iris_dataset['DESCR']}")
"""
print("Назви відповідей:{}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data:{}".format(iris_dataset['data'].shape))
print(F"Data count:{len(iris_dataset['data'])}")
print(F"First 5 samples:\n {iris_dataset['data'][:5]}")
print("Тип масиву target:{}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))
"""



