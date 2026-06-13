import gradio as gr
import pickle
import numpy as np
from realization import Distributor
from test import Test
import os

class App:
    def __init__(self):
        self.model_info = None

    def load_model(self):
        if self.model_info is None:
            with open('best_model.pkl', 'rb') as f:
                self.model_info = pickle.load(f)
        return self.model_info

    def train_model(self):
        from realization import Distributor

        try:
            self.distributor = Distributor()
            self.distributor.preporation_init()
            self.distributor.fit_pred_model()
            self.distributor.analysis()

            self.model_info = None

            return (
                "Обучение завершено!",
                "comparison_metrics.png",
                "confusion_matrix_hard.png",
                "confusion_matrix_simple.png"
            )
        except Exception as e:
            return (f"Ошибка: {str(e)}", None, None, None)

    def test_model(self):
        if not os.path.exists('best_model.pkl'):
            return "Модель не найдена. Сначала обучите модель во вкладке 'Обучение'."
        try:
            self.distributor = Distributor()
            self.distributor.preporation_init()

            tests = Test()
            tests.load_model('best_model.pkl')
            tests.test_data(self.distributor.X_test, self.distributor.y_test)

            return tests.print_report()
        except Exception as e:
            return f"Ошибка тестирования: {str(e)}"

    def predict_genre(self,description):
        if not description or not description.strip():
            return "Введите описание фильма"
        try:
            model_info = self.load_model()
        except FileNotFoundError:
            return "Модель не найдена. Сначала обучите модель."
        X = model_info['tfidf'].transform([description])
        X_full = np.hstack([np.zeros((1, 2)), X.toarray()])
        model = model_info['model']
        if model_info['model_type'] == 'simple':
            genre = model.predict_simple(X_full)[0]
        else:
            genre = model.predict_hard(X_full)[0]
        return genre

    def launch(self):
        with gr.Blocks(title="Классификация жанров фильмов") as demo:
            gr.Markdown("Классификация жанров фильмов")
            gr.Markdown("Определите жанр фильма по его описанию")

            with gr.Tab("Предсказание"):
                gr.Markdown("Введите описание фильма на английском языке")

                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            label="Описание фильма",
                            placeholder="A young wizard discovers his magical heritage and attends a school of witchcraft and wizardry...",
                            lines=4
                        )
                        predict_btn = gr.Button("Определить жанр", variant="primary")

                    with gr.Column(scale=1):
                        output_text = gr.Textbox(label="Предсказанный жанр")

                predict_btn.click(
                    fn=self.predict_genre,
                    inputs=input_text,
                    outputs=output_text
                )

                gr.Examples(
                    examples=[
                        "A group of heroes must destroy a powerful ring to save the world",
                        "A detective investigates a series of mysterious murders in a small town",
                        "Two strangers fall in love during a trip across Europe",
                        "A team of explorers discovers ancient ruins with deadly secrets",
                        "A retired agent is forced back into action to rescue his daughter"
                    ],
                    inputs=input_text,
                    label="Примеры описаний"
                )

            with gr.Tab("Обучение"):
                train_btn = gr.Button("Запустить обучение")
                train_output = gr.Textbox(label="Результат")

                train_btn.click(
                    fn=self.train_model,
                    outputs=[train_output,
                             gr.Image("comparison_metrics.png"),
                             gr.Image("confusion_matrix_hard.png"),
                             gr.Image("confusion_matrix_simple.png")]
                )

            with gr.Tab("Тестирование"):
                gr.Markdown("Протестируйте сохранённую модель на отложенных данных.")

                test_btn = gr.Button("Запустить тест", variant="primary")
                test_output = gr.Textbox(label="Результат тестирования", lines=5)

                test_btn.click(
                    fn=self.test_model,
                    outputs=test_output
                )
            gr.Markdown("Модель обучена на данных Movies Genre Description")

        demo.launch(inbrowser=True)
