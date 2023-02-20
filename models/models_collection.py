from sklearn import ensemble #ансамбли



#Создаем объект класса случайный лес

def ModelRandomForest(config):
    """Метод формирования модели
    """
    rf = ensemble.RandomForestClassifier(
        n_estimators=int(config.n_estimators), #число деревьев
        criterion=config.criterion, #критерий эффективности
        max_depth=int(config.max_depth), #максимальная глубина дерева
        min_samples_leaf = int(config.min_samples_leaf), # Минимальное число объектов в листе
        random_state=int(config.random_seed) #генератор случайных чисел
    )

    return rf