import pytest
import sys
from pathlib import Path

# Добавляем корень проекта в путь
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

predict_function = None

# Импортируем модель
if predict_function is None:
    try:
        # Загружаем модель напрямую
        import pickle
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        model_path = root_dir / 'best_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # Если модель - это словарь с векторизатором и классификатором
            if isinstance(model_data, dict):
                vectorizer = model_data.get('vectorizer')
                classifier = model_data.get('classifier')


                def simple_predict(text):
                    if vectorizer and classifier:
                        X = vectorizer.transform([text])
                        return classifier.predict(X)[0]
                    return "unknown"


                predict_function = simple_predict
                print("Загружена модель напрямую из best_model.pkl")
    except Exception as e:
        print(f"Не удалось загрузить модель напрямую: {e}")


# ===== ТЕСТ 1: Проверка работы предсказания =====
def test_prediction_runs():
    """Проверяет, что функция предсказания работает на корректном примере"""
    # Если функция не найдена, тест пропускаем
    if predict_function is None:
        pytest.skip("Не найдена функция предсказания. Проверьте файлы realization.py и main.py")

    try:
        # Короткое описание фильма
        sample_text = "A young wizard goes to magic school and fights evil"
        result = predict_function(sample_text)

        # Проверяем, что результат не None и не пустой
        assert result is not None, "Функция вернула None"
        assert result != "", "Функция вернула пустую строку"

        print(f"Успешно! Предсказание: {result}")

    except Exception as e:
        pytest.fail(f"Ошибка при вызове функции предсказания: {e}")


# ===== ТЕСТ 2: Проверка формата ответа =====
def test_prediction_format():
    """Проверяет, что ответ возвращается в правильном формате (строка)"""
    if predict_function is None:
        pytest.skip("Не найдена функция предсказания")

    sample_text = "Two robots fight in a futuristic city"
    result = predict_function(sample_text)

    # Проверяем тип результата
    assert isinstance(result, str), f"Ожидалась строка (str), получено {type(result).__name__}"

    # Проверяем, что строка не слишком длинная (разумное имя жанра)
    assert len(result) < 100, f"Результат слишком длинный: {len(result)} символов"

    print(f"Формат ответа корректный: {result}")


# ===== ТЕСТ 3: Проверка веб-приложения =====
def test_app_imports():
    """Проверяет, что файл веб-приложения может быть импортирован без ошибок"""
    try:
        # Пробуем импортировать app.py
        import app
        assert hasattr(app, '__file__'), "Модуль app загружен некорректно"

        # Проверяем, есть ли там функция запуска
        if hasattr(app, 'main'):
            print("В app.py найдена функция main()")
        elif hasattr(app, 'app'):
            print("В app.py найдена переменная app (Flask/Streamlit)")
        else:
            print("app.py загружен, но структура нестандартная")

    except ImportError as e:
        pytest.fail(f"Не удалось импортировать app.py: {e}")
    except Exception as e:
        pytest.fail(f"Ошибка при импорте app.py: {e}")


# ===== ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: Проверка наличия модели =====
def test_model_exists():
    """Проверяет, что файл обученной модели существует"""
    model_path = root_dir / 'best_model.pkl'
    assert model_path.exists(), f"Файл модели не найден по пути: {model_path}"

    # Проверяем, что файл не пустой
    assert model_path.stat().st_size > 1000, "Файл модели слишком маленький (возможно, поврежден)"

    print(f"Модель найдена: {model_path}")