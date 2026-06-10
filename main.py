# This is a sample Python script.
from realization import *
from pathlib import Path
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.





# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    test_reliz = Distributor()
    test_reliz.preporation_init()
    print("Предобработка завершена")
    if Path('best_model.pkl').exists()==False:
        test_reliz.fit_pred_model()
        print("Предсказание завершено завершена")
        test_reliz.analysis()
        print("Обучение завершено")
    test_reliz.test_data()
    print("Тестирование завершено")

