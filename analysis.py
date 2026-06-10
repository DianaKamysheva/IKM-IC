from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,f1_score
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class Analyzer:
    def __init__(self,cv,tfidf):
        self.cv_results = cv
        self.all_true = None
        self.all_pred_simple = None
        self.all_pred_hard_encoded = None
        self.classes = None
        self.metrics_simple = None
        self.metrics_hard = None
        self.tfidf=tfidf


    def metrics(self,label_encoder):
        all_true = []
        all_pred_simple = []
        all_pred_hard = []

        for fold in self.cv_results:
            all_true.extend(fold['y_true'])
            all_pred_simple.extend(fold['y_pred_simple'])
            all_pred_hard.extend(fold['y_pred_hard'])
        all_pred_hard_encoded = label_encoder.transform(all_pred_hard)
        classes = label_encoder.classes_
        precision_simple, recall_simple, f1_simple, _ = precision_recall_fscore_support(
            all_true, all_pred_simple, average=None, zero_division=0
        )
        precision_hard, recall_hard, f1_hard, _ = precision_recall_fscore_support(
            all_true, all_pred_hard_encoded, average=None, zero_division=0
        )
        self.metrics_simple = {
            'precision': dict(zip(classes, precision_simple)),
            'recall': dict(zip(classes, recall_simple)),
            'f1': dict(zip(classes, f1_simple))
        }
        self.metrics_hard = {
            'precision': dict(zip(classes, precision_hard)),
            'recall': dict(zip(classes, recall_hard)),
            'f1': dict(zip(classes, f1_hard))
        }
        self.all_true = all_true
        self.all_pred_simple = all_pred_simple
        self.all_pred_hard_encoded = all_pred_hard_encoded
        self.classes = classes

        avg_f1_simple = f1_simple.mean()
        avg_f1_hard = f1_hard.mean()
        best_fold_idx = 0
        best_fold_f1 = 0
        if avg_f1_hard >= avg_f1_simple:
            for idx, fold in enumerate(self.cv_results):
                fold_hard_encoded = label_encoder.transform(fold['y_pred_hard'])
                fold_f1 = f1_score(
                    fold['y_true'],
                    fold_hard_encoded,
                    average='weighted',
                    zero_division=0
                )
                if fold_f1 > best_fold_f1:
                    best_fold_f1 = fold_f1
                    best_fold_idx = idx

            self.best_model_info = {
                'model_type': 'hard',
                'model': self.cv_results[best_fold_idx]['hard_model'],
                'fold': best_fold_idx + 1,
                'f1_score': best_fold_f1,
                'label_encoder': label_encoder
            }
        else:
            for idx, fold in enumerate(self.cv_results):
                fold_f1 = f1_score(
                    fold['y_true'],
                    fold['y_pred_simple'],
                    average='weighted',
                    zero_division=0
                )
                if fold_f1 > best_fold_f1:
                    best_fold_f1 = fold_f1
                    best_fold_idx = idx

            self.best_model_info = {
                'model_type': 'simple',
                'model': self.cv_results[best_fold_idx]['simple_model'],
                'fold': best_fold_idx + 1,
                'f1_score': best_fold_f1,
                'label_encoder': label_encoder,
                'tfidf': self.tfidf
            }


    def save_best_model(self, filepath='best_model.pkl'):
        if not hasattr(self, 'best_model_info') or self.best_model_info is None:
            return "Сначала запустите metrics()"

        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model_info, f)

        report=[]
        report.append(f"Модель сохранена: {filepath}")
        report.append(f"Тип: {self.best_model_info['model_type']}")
        report.append(f"F1: {self.best_model_info['f1_score']:.4f}")
        return "\n".join(report)



    def plot(self):
        comparison_data = []
        for class_name in self.classes:
            comparison_data.append({
                'Класс': class_name,
                'F1 Simple': f"{self.metrics_simple['f1'][class_name]:.3f}",
                'F1 Hard': f"{self.metrics_hard['f1'][class_name]:.3f}",
                'Разница': f"{self.metrics_hard['f1'][class_name] - self.metrics_simple['f1'][class_name]:.3f}"
            })
        df_comparison = pd.DataFrame(comparison_data)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        #Сравнение F1-score по классам
        x = np.arange(len(self.classes))
        width = 0.35

        f1_simple_vals = [self.metrics_simple['f1'][c] for c in self.classes]
        f1_hard_vals = [self.metrics_hard['f1'][c] for c in self.classes]

        axes[0, 0].bar(x - width / 2, f1_simple_vals, width, label='Simple', alpha=0.7)
        axes[0, 0].bar(x + width / 2, f1_hard_vals, width, label='Hard', alpha=0.7)
        axes[0, 0].set_xlabel('Классы', fontsize=10, labelpad=10)
        axes[0, 0].set_ylabel('F1-score', fontsize=10)
        axes[0, 0].set_title('Сравнение F1-score: Simple vs Hard', fontsize=10)
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(self.classes, rotation=45)
        axes[0, 0].legend()

        #Сравнение precision
        prec_simple_vals = [self.metrics_simple['precision'][c] for c in self.classes]
        prec_hard_vals = [self.metrics_hard['precision'][c] for c in self.classes]
        axes[0, 1].bar(x - width / 2, prec_simple_vals, width, label='Simple', alpha=0.7)
        axes[0, 1].bar(x + width / 2, prec_hard_vals, width, label='Hard', alpha=0.7)
        axes[0, 1].set_xlabel('Классы', fontsize=10, labelpad=10)
        axes[0, 1].set_ylabel('Precision', fontsize=10)
        axes[0, 1].set_title('Сравнение Precision: Simple vs Hard', fontsize=10)
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(self.classes, rotation=45)
        axes[0, 1].legend()

        #Сравнение recall
        rec_simple_vals = [self.metrics_simple['recall'][c] for c in self.classes]
        rec_hard_vals = [self.metrics_hard['recall'][c] for c in self.classes]

        axes[1, 0].bar(x - width / 2, rec_simple_vals, width, label='Simple', alpha=0.7)
        axes[1, 0].bar(x + width / 2, rec_hard_vals, width, label='Hard', alpha=0.7)
        axes[1, 0].set_xlabel('Классы', fontsize=10, labelpad=10)
        axes[1, 0].set_ylabel('Recall', fontsize=10)
        axes[1, 0].set_title('Сравнение Recall: Simple vs Hard', fontsize=10)
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(self.classes, rotation=45)
        axes[1, 0].legend()

        #Где улучшилось/ухудшилось
        diff = [f1_hard_vals[i] - f1_simple_vals[i] for i in range(len(self.classes))]
        colors = ['green' if d > 0 else 'red' for d in diff]

        axes[1, 1].barh(self.classes, diff, color=colors, alpha=0.7)
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].set_xlabel('Изменение F1-score')
        axes[1, 1].set_title('Где Hard лучше/хуже Simple')

        plt.tight_layout()
        plt.show()

        print(df_comparison.to_string(index=False))

        return df_comparison

    def show_error_analysis(self, prediction_type='hard'):
        if prediction_type == 'hard':
            y_pred = self.all_pred_hard_encoded
            title = 'Hard predictions'
        else:
            y_pred = self.all_pred_simple
            title = 'Simple predictions'

        y_true = np.array(self.all_true)
        cm = confusion_matrix(y_true, y_pred)
        errors_info = []
        #Поиск главной ошибки
        for true_class in range(len(self.classes)):
            for pred_class in range(len(self.classes)):
                if true_class != pred_class and cm[true_class, pred_class] > 0:
                    errors_info.append({
                        'Истина': self.classes[true_class],
                        'Предсказано': self.classes[pred_class],
                        'Количество': cm[true_class, pred_class]
                    })

        errors_df = pd.DataFrame(errors_info)
        errors_df = errors_df.sort_values('Количество', ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        annot_array = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] > 0:
                    annot_array[i, j] = str(cm[i, j])
                else:
                    annot_array[i, j] = ''

        sns.heatmap(cm,
                    annot=annot_array,
                    fmt='',
                    cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes,
                    ax=axes[0],
                    annot_kws={'size': 7},
                    cbar_kws={'shrink': 0.8})

        axes[0].set_title(f'Confusion Matrix - {title}')
        axes[0].set_xlabel('Предсказано')
        axes[0].set_ylabel('Истина')

        top_errors = errors_df.head(10)
        labels = [f"{e['Истина']} → {e['Предсказано']}" for _, e in top_errors.iterrows()]
        axes[1].barh(labels, top_errors['Количество'], color='salmon')
        axes[1].set_xlabel('Количество ошибок')
        axes[1].set_title(f'Топ-10 ошибок - {title}')
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.show()

        return errors_df

