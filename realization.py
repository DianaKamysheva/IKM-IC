from train import *
from analysis import *
from test import *
import copy

class Distributor:
    def __init__(self):
        self.preparation = None
        self.analyzer = None
        self.tests=None
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.cv_results = []
        self.label_encoder=None
        self.tfidf=None

    def preporation_init(self):
        print("Начата предобработка")
        self.preparation = Preparation("Movies_Genre_Description.csv")
        self.X_train, self.X_test, self.y_train, self.y_test = self.preparation.catigorToInt()
        self.tfidf=self.preparation.get_tfidf()

    def fit_pred_model(self):
        print("Начато предсказание")
        self.label_encoder = self.preparation.get_label_encoder()
        splits = FitPredict(self.label_encoder).cross_val(self.X_train, self.y_train)

        # Создаём ОДИН экземпляр FitPredict
        fit = FitPredict(self.label_encoder)
        for split in splits:
            print("Этап +")
            fit.genre_marker = {}
            fit.default_genre = None

            fit.fit_simple(split['X_train_fold'], split['y_train_fold'])
            y_pred_simple = fit.predict_simple(split['X_val_fold'])
            fit.fit_hard(split['X_train_fold'], split['y_train_fold'])
            y_pred_hard = fit.predict_hard(split['X_val_fold'])

            self.cv_results.append({
                'y_true': split['y_val_fold'],
                'y_pred_simple': y_pred_simple,
                'y_pred_hard': y_pred_hard,
                'simple_model': copy.deepcopy(fit),
                'hard_model': copy.deepcopy(fit)
            })

    def analysis(self):
        self.analyzer = Analyzer(self.cv_results,self.tfidf)
        self.analyzer.metrics(self.label_encoder)
        self.analyzer.plot()
        self.analyzer.show_error_analysis()
        self.analyzer.show_error_analysis('simple')
        self.analyzer.save_best_model()

    def test_data(self):
        self.tests=Test()
        self.tests.test_data(self.X_test, self.y_test)
        self.tests.print_report()
