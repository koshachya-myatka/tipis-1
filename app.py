# pip install flask
from flask import Flask, request, render_template, jsonify, send_file, session
import os
import io
import base64

from calculator import Calculator

app = Flask(__name__)

CALCULATOR = None


# инициализация калькулятора со случайными значениями
def init_calculator():
    global CALCULATOR
    CALCULATOR = Calculator()
    return CALCULATOR.parameters


# обновление параметров калькулятора из данных формы
def update_calculator_from_form(form_data):
    global CALCULATOR
    CALCULATOR.parameters = {}
    for key, value in form_data.items():
        if value and value != "":
            try:
                if "." in value:
                    CALCULATOR.parameters[key] = float(value)
                else:
                    CALCULATOR.parameters[key] = int(value)
            except ValueError:
                CALCULATOR.parameters[key] = value
    return CALCULATOR.parameters


# главная страница и внедрение модели калькулятора
@app.route("/")
def index():
    global CALCULATOR
    if CALCULATOR is None:
        parameters = init_calculator()
    else:
        parameters = CALCULATOR.parameters
    return render_template("index.html", parameters=parameters)


# очистка всех значений
@app.route("/clear", methods=["POST"])
def clear_values():
    global CALCULATOR
    CALCULATOR = None
    return jsonify({"status": "success"})


# случайные значения
@app.route("/random", methods=["POST"])
def random_values():
    parameters = init_calculator()
    return jsonify({"status": "success", "parameters": parameters})


# построение диаграмм
@app.route("/plot", methods=["POST"])
def plot_diagrams():
    global CALCULATOR
    form_data = request.json

    # проверка на заполненность всех полей
    empty_fields = []
    for key, value in form_data.items():
        if value == "":
            empty_fields.append(key)
    if empty_fields:
        return jsonify(
            {
                "status": "error",
                "message": "Не все поля заполнены",
                "empty_fields": empty_fields,
            }
        )

    # обновляем калькулятор из формы
    update_calculator_from_form(form_data)

    # создаем картинки диаграмм и выводим их
    try:
        time_series_b64, radar_b64 = CALCULATOR.plot_all_results()
        return jsonify(
            {
                "status": "success",
                "time_series": time_series_b64,
                "radar_charts": radar_b64,
            }
        )
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"Ошибка при построении графиков: {str(e)}"}
        )
    

if __name__ == "__main__":
    app.run(port=8000, debug=True)
