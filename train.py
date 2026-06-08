import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from collections import Counter


class Preparation:
    def __init__(self, data_name):
        self.data = pd.read_csv(data_name)

        min_samples = 2
        genre_counts = self.data['GENRE'].value_counts()
        rare_genres = genre_counts[genre_counts < min_samples].index

        if len(rare_genres) > 0:
            mask = ~self.data['GENRE'].isin(rare_genres)
            self.data = self.data[mask].reset_index(drop=True)

        self.X = self.data.drop(columns=['GENRE'])
        self.y = self.data['GENRE']
        self.label_encoder = None

    def get_label_encoder(self):
        return self.label_encoder

    def catigorToInt(self):
        # Кодируем целевую переменную
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        # Кодируем TITLE
        self.X["TITLE"] = LabelEncoder().fit_transform(self.X["TITLE"].astype(str))

        # Обрабатываем DATE
        years = self.X["DATE"].replace('????', np.nan)
        years = pd.to_numeric(years, errors='coerce')
        mean_year = years.mean()
        years_filled = years.fillna(mean_year)
        scaler = StandardScaler()
        years_after = scaler.fit_transform(years_filled.values.reshape(-1, 1)).flatten()
        self.X["DATE"] = years_after

        print("TF-IDF")
        # TF-IDF
        tfidf = TfidfVectorizer(
            max_features=2500,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8
        )
        texts = self.X["DESCRIPTION"].fillna('')
        tfidf_matrix = tfidf.fit_transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'word_{i}' for i in range(tfidf_matrix.shape[1])]
        )

        self.X = self.X.drop(columns=["DESCRIPTION"])
        self.X = pd.concat([self.X, tfidf_df], axis=1)

        return train_test_split(
            self.X, self.y,
            test_size=0.3,
            random_state=42,
            stratify=self.y
        )

class FitPredict:
    def __init__(self,label_encoder):
        self.genre_marker = {}
        self.default_genre = None
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=42)
        )
        self.label_encoder = label_encoder
        self.genre_thresholds={}

    def cross_val(self,X,y):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = []
        for train_idx, val_idx in cv.split(X, y):
            splits.append({
                'X_train_fold': X.iloc[train_idx],
                'y_train_fold': y[train_idx],
                'X_val_fold' : X.iloc[val_idx],
                'y_val_fold' : y[val_idx],
            })
        return splits

    def _prepare_prediction_matrix(self):
        """
        Подготовка матрицы весов для быстрого предсказания.
        Вместо цикла по жанрам — одно матричное умножение.
        """
        genres = list(self.genre_weights.keys())

        # Создаём матрицу [n_genres × n_features]
        self._weights_matrix = np.zeros((len(genres), len(self.genre_weights[genres[0]])),
                                        dtype=np.float32)
        self._genre_list = genres

        for i, genre in enumerate(genres):
            self._weights_matrix[i] = self.genre_weights[genre]


    def fit_simple(self,X,y):
        if hasattr(X, 'toarray'):
            X_array = X.toarray().astype(np.float32)
        elif hasattr(X, 'values'):
            X_array = X.values.astype(np.float32)
        else:
            X_array = np.array(X, dtype=np.float32)

        genres = np.unique(y)
        n_samples, n_features = X_array.shape

        doc_frequency = (X_array > 0).sum(axis=0) / n_samples
        frequent_words_mask = doc_frequency > 0.7

        genre_means = np.zeros((len(genres), n_features))
        for i, genre in enumerate(genres):
            genre_means[i] = X_array[y == genre].mean(axis=0)

        variance_across_genres = genre_means.var(axis=0)

        useless_words_mask = frequent_words_mask | (variance_across_genres < 1e-6)

        print(f"Отфильтровано {useless_words_mask.sum()} бесполезных слов")

        self.genre_weights = {}

        for genre in genres:
            genre_mask = (y == genre)

            genre_mean = X_array[genre_mask].mean(axis=0)
            other_mean = X_array[~genre_mask].mean(axis=0)

            weights = (genre_mean - other_mean).astype(np.float32)
            weights = np.maximum(weights, 0)

            # Обнуляем бесполезные слова
            weights[useless_words_mask] = 0

            # L2-нормализация (косинусное сходство)
            l2_norm = np.linalg.norm(weights)
            if l2_norm > 0:
                weights = weights / l2_norm

            self.genre_weights[genre] = weights

        self.default_genre = Counter(y).most_common(1)[0][0]
        self._prepare_prediction_matrix()




    def predict_simple(self,X):
        if hasattr(X, 'values'):
            X_array = X.values.astype(np.float32)
        elif hasattr(X, 'toarray'):
            X_array = X.toarray().astype(np.float32)
        else:
            X_array = np.array(X, dtype=np.float32)

        n_samples = len(X_array)

        scores = X_array @ self._weights_matrix.T  # [n_samples × n_genres]

        best_genre_indices = np.argmax(scores, axis=1)
        max_scores = np.max(scores, axis=1)

        predictions = np.full(n_samples, self.default_genre, dtype=object)

        mask = max_scores > 0
        predictions[mask] = [self._genre_list[idx] for idx in best_genre_indices[mask]]

        return predictions.tolist()


    def fit_hard(self,X,y):
        self.model.fit(X, y)
        return self

    def predict_hard(self,X):
        predictions_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions_encoded)

