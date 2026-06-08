import pickle
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

class Test:

    def __init__(self):
        self.model_info = None
        self.test_results = None

    def load_model(self,filepath='best_model.pkl'):
        with open(filepath, 'rb') as f:
            self.model_info = pickle.load(f)
        print(f"Модель загружена: {self.model_info['model_type']}")

    def test_data(self, X_test, y_test):
        if self.model_info is None:
            self.load_model()
        model = self.model_info['model']
        model_type = self.model_info['model_type']
        label_encoder = self.model_info['label_encoder']

        if model_type == 'simple':
            y_pred = model.predict(X_test)
        else:
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)

        if isinstance(y_pred[0], str) and not isinstance(y_test[0], str):
            y_test = label_encoder.inverse_transform(y_test)

        y_test = [str(y) for y in y_test]
        y_pred = [str(y) for y in y_pred]

        test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        classes = np.unique(np.concatenate([y_test, y_pred]))
        max_error = 0
        error_pair = (None, None)

        for i, true_c in enumerate(classes):
            for j, pred_c in enumerate(classes):
                if i != j and cm[i, j] > max_error:
                    max_error = cm[i, j]
                    error_pair = (true_c, pred_c)

        self.test_results = {
            'y_true': y_test,
            'y_pred': y_pred,
            'f1_score': test_f1,
            'confusion_matrix': cm,
            'main_error': error_pair,
            'main_error_count': max_error
        }

        return self.test_results

    def print_report(self):
        if self.test_results is None:
            print("Сначала запустите test()")
            return

        model_type = self.model_info['model_type'].upper()
        f1 = self.test_results['f1_score']
        err_from, err_to = self.test_results['main_error']
        err_count = self.test_results['main_error_count']

        print("ИТОГОВЫЙ ОТЧЁТ")
        print(f"Лучшая модель — {model_type}-модель.")
        print(f"Её ключевая метрика на новых данных — {f1:.4f}.")
        print(f"Чаще всего она путает '{err_from}' и '{err_to}' ({err_count} раз).")