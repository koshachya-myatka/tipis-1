# pip install flask
from flask import Flask, request, render_template, jsonify

from calculator import Calculator
from calculator2 import Calculator2

app = Flask(__name__)

CALCULATOR = None
CALCULATOR2 = None


# инициализация калькулятора со случайными значениями
def init_calculator():
    global CALCULATOR
    CALCULATOR = Calculator()
    return CALCULATOR.parameters


# обновление параметров калькулятора из данных формы
def update_calculator_from_form(form_data):
    global CALCULATOR
    # Инициализируем калькулятор, если он еще не создан
    if CALCULATOR is None:
        init_calculator()

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


# инициализация калькулятора2 со случайными значениями
def init_calculator2():
    global CALCULATOR2
    CALCULATOR2 = Calculator2()
    return CALCULATOR2.parameters


# обновление параметров калькулятора2 из данных формы
def update_calculator2_from_form(form_data):
    global CALCULATOR2
    # Инициализируем калькулятор, если он еще не создан
    if CALCULATOR2 is None:
        init_calculator2()

    CALCULATOR2.parameters = {}
    for key, value in form_data.items():
        if value and value != "":
            try:
                if "." in value:
                    CALCULATOR2.parameters[key] = float(value)
                else:
                    CALCULATOR2.parameters[key] = int(value)
            except ValueError:
                CALCULATOR2.parameters[key] = value

    # Убедимся, что все необходимые X параметры присутствуют
    for i in range(1, 19):
        param_name = f"X{i}"
        if param_name not in CALCULATOR2.parameters:
            # Если параметр отсутствует, установим значение по умолчанию
            min_val, max_val, _ = CALCULATOR2.pR[param_name]
            CALCULATOR2.parameters[param_name] = (min_val + max_val) / 2

    # Также убедимся, что присутствуют min/max значения
    for i in range(1, 19):
        param_name = f"X{i}"
        min_name = f"{param_name}_min"
        max_name = f"{param_name}_max"

        if min_name not in CALCULATOR2.parameters:
            min_val, max_val, _ = CALCULATOR2.pR[param_name]
            CALCULATOR2.parameters[min_name] = min_val

        if max_name not in CALCULATOR2.parameters:
            min_val, max_val, _ = CALCULATOR2.pR[param_name]
            CALCULATOR2.parameters[max_name] = max_val

    # Обновляем нормализованные параметры
    CALCULATOR2.parameters_norm = CALCULATOR2._get_normalized(CALCULATOR2.parameters)
    return CALCULATOR2.parameters


# главная страница и внедрение модели калькулятора2
@app.route("/2")
def index2():
    global CALCULATOR2
    if CALCULATOR2 is None:
        parameters = init_calculator2()
    else:
        parameters = CALCULATOR2.parameters
    return render_template("index2.html", parameters=parameters)


# очистка всех значений
@app.route("/clear2", methods=["POST"])
def clear_values2():
    global CALCULATOR2
    CALCULATOR2 = None
    return jsonify({"status": "success"})


# случайные значения
@app.route("/random2", methods=["POST"])
def random_values2():
    parameters = init_calculator2()
    return jsonify({"status": "success", "parameters": parameters})


# построение диаграмм
@app.route("/plot2", methods=["POST"])
def plot_diagrams2():
    global CALCULATOR2
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
    update_calculator2_from_form(form_data)

    # создаем картинки диаграмм и выводим их
    try:
        # Пересчитываем решение системы с новыми параметрами
        CALCULATOR2.solve_system()

        time_series_b64 = CALCULATOR2.plot_time_series()
        radar_b64 = CALCULATOR2.plot_radar_charts()

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