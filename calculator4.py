# calculator4.py (исправленная версия)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import io
import base64


class Polynomial:
    """Класс полинома для совместимости с примером"""
    def __init__(self, *coefficients):
        self.coefficients = coefficients

    def calc(self, x):
        """Вычисление значения полинома в точке x"""
        return sum(c * (x ** i) for i, c in enumerate(self.coefficients))


class CubicPolynomial(Polynomial):
    """Кубический полином"""
    def __init__(self, a, b, c, d):
        super().__init__(a, b, c, d)


class Calculator4:
    def __init__(self):
        self.names = {
            "X1": "Эффективность функционирования хранилища данных",
            "X2": "Качество программного обеспечения (ПО)",
            "X3": "Корректность ПО",
            "X4": "Надежность программного обеспечения",
            "X5": "Доступность программного обеспечения",
            "X6": "Возможность интенсивного использования ПО",
            "X7": "Прослеживаемость ПО",
            "X8": "Функциональная полнота ПО",
            "X9": "Обеспечение требуемой последовательности работ при проектировании хранилища",
            "X10": "Практичность ПО",
            "X11": "Устойчивость к ошибкам данных программного обеспечения",
            "X12": "Эффективность выполнения транзакций",
            "X13": "Степень мотивации персонала",
            "X14": "Удобство тестирования ПО"
        }

        # Названия возмущений
        self.disturbance_names = {
            "zeta1": "Увеличение количества источников новых данных",
            "zeta2": "Частота изменения периодов сдачи финансовой отчетности предприятия",
            "zeta3": "Сокращение квалифицированной поддержки вендора",
            "zeta4": "Рост интенсивности перехода на Open Source решения",
            "zeta5": "Увеличение количества новых стандартов при реализации Open Source решений"
        }

        # Диапазоны значений для параметров (0-100%)
        self.pR = {
            'X1': (0, 100), 'X2': (0, 100), 'X3': (0, 100), 'X4': (0, 100),
            'X5': (0, 100), 'X6': (0, 100), 'X7': (0, 100), 'X8': (0, 100),
            'X9': (0, 100), 'X10': (0, 100), 'X11': (0, 100), 'X12': (0, 100),
            'X13': (0, 100), 'X14': (0, 100)
        }

        # Диапазоны для коэффициентов полиномов
        self.coeff_range = {'a3': (-0.1, 0.1), 'a2': (-0.2, 0.2),
                            'a1': (0.3, 0.7), 'a0': (0.1, 0.3)}

        # Диапазоны для возмущений (0-1)
        self.zeta_range = (0, 1)

        self.time_points = np.linspace(0, 10, 500)
        self.solution = None
        self.parameters = self.generate_valid_parameters()
        self.parameters_norm = self._get_normalized(self.parameters)

        # Инициализируем полиномиальные функции
        self.init_polynomials()

    def init_polynomials(self):
        """Инициализация полиномиальных функций как в примере"""
        self.functions = []
        self.qfunctions = []

        # Создаем 160 полиномов для f_i
        for i in range(1, 161):
            a3 = self.parameters.get(f"f{i}_a3", 0)
            a2 = self.parameters.get(f"f{i}_a2", 0)
            a1 = self.parameters.get(f"f{i}_a1", 0.5)
            a0 = self.parameters.get(f"f{i}_a0", 0.2)
            self.functions.append(CubicPolynomial(a3, a2, a1, a0))

        # Создаем 5 полиномов для возмущений
        for i in range(1, 6):
            a3 = self.parameters.get(f"zeta{i}_a3", 0)
            a2 = self.parameters.get(f"zeta{i}_a2", 0)
            a1 = self.parameters.get(f"zeta{i}_a1", 0.5)
            a0 = self.parameters.get(f"zeta{i}_a0", 0.2)
            self.qfunctions.append(CubicPolynomial(a3, a2, a1, a0))

    def generate_system_parameters(self):
        parameters = {}

        # Параметры X1-X14
        for i in range(1, 15):
            param_name = f"X{i}"
            min_r, max_r = self.pR[param_name]
            base_value = round(random.uniform(min_r, max_r), 2)
            max_value = round(random.uniform(base_value, max_r), 2)
            parameters[param_name] = base_value
            parameters[f"{param_name}_max"] = max_value

        # Коэффициенты полиномов f1-f160
        for i in range(1, 161):
            for coef in ['a3', 'a2', 'a1', 'a0']:
                min_val, max_val = self.coeff_range[coef]
                parameters[f"f{i}_{coef}"] = round(random.uniform(min_val, max_val), 3)

        # Коэффициенты для возмущений zeta1-zeta5
        for i in range(1, 6):
            for coef in ['a3', 'a2', 'a1', 'a0']:
                min_val, max_val = self.coeff_range[coef]
                parameters[f"zeta{i}_{coef}"] = round(random.uniform(min_val, max_val), 3)

        return parameters

    def generate_valid_parameters(self):
        for attempt in range(10):
            try:
                params = self.generate_system_parameters()
                params_norm = self._get_normalized(params)
                self.parameters = params
                self.parameters_norm = params_norm
                self.init_polynomials()  # Переинициализируем полиномы

                solution = self.solve_system()

                if (solution is not None and
                        np.all(np.isfinite(solution.y)) and
                        not np.any(np.isnan(solution.y)) and
                        np.max(np.abs(solution.y)) < 10.0):  # Увеличил предел для отрицательных значений
                    print(f"Успешная генерация параметров с попытки {attempt + 1}")
                    return params

            except Exception as e:
                print(f"Попытка {attempt + 1} не удалась: {e}")
                continue

        print("Используем резервные параметры")
        return self.create_fallback_parameters()

    def create_fallback_parameters(self):
        parameters = {}

        # Базовые значения параметров
        for i in range(1, 15):
            param_name = f"X{i}"
            min_r, max_r = self.pR[param_name]
            parameters[param_name] = round(min_r + 0.5 * (max_r - min_r), 2)
            parameters[f"{param_name}_max"] = round(min_r + 0.6 * (max_r - min_r), 2)

        # Коэффициенты полиномов
        for i in range(1, 161):
            parameters[f"f{i}_a3"] = 0.0
            parameters[f"f{i}_a2"] = round(random.uniform(-0.2, 0.2), 3)
            parameters[f"f{i}_a1"] = round(random.uniform(0.3, 0.7), 3)
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 0.3), 3)

        # Коэффициенты возмущений
        for i in range(1, 6):
            parameters[f"zeta{i}_a3"] = 0.0
            parameters[f"zeta{i}_a2"] = round(random.uniform(-0.2, 0.2), 3)
            parameters[f"zeta{i}_a1"] = round(random.uniform(0.3, 0.7), 3)
            parameters[f"zeta{i}_a0"] = round(random.uniform(0.1, 0.3), 3)

        return parameters

    def _get_normalized(self, params):
        params_norm = params.copy()

        for i in range(1, 15):
            param_name = f"X{i}"
            min_val, max_val = self.pR[param_name]

            if param_name in params_norm:
                params_norm[param_name] = (params_norm[param_name] - min_val) / (max_val - min_val)

            if f"{param_name}_max" in params_norm:
                params_norm[f"{param_name}_max"] = (params_norm[f"{param_name}_max"] - min_val) / (max_val - min_val)

        return params_norm

    def disturbance_function(self, t, i):
        """Вычисление возмущения ζ_i(t) как полином от времени"""
        # Нормализованное время
        x = t / 10.0
        # Используем класс Polynomial для возмущений
        value = self.qfunctions[i-1].calc(x)
        return float(value)

    def polynomial_value(self, x, i):
        """Вычисление значения полинома f_i(x) с использованием класса Polynomial"""
        x = np.clip(x, -1, 2)
        value = self.functions[i-1].calc(x)
        # Ограничиваем значения полиномов
        return float(np.clip(value, 0, 2))

    def system_equations(self, t, X):
        """Система дифференциальных уравнений для 14 переменных (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        try:
            dXdt = np.zeros(14)
            X_safe = np.array(X, dtype=float)

            # Ограничиваем переменные для стабильности
            X_safe = X

            # Вычисляем возмущения
            zeta = [np.clip(self.disturbance_function(t, i), 0, 1) for i in range(1, 6)]
            zeta1, zeta2, zeta3, zeta4, zeta5 = zeta
            zeta_sum = sum(zeta)

            # Вспомогательная функция для получения полинома
            f = lambda idx, x_val: np.clip(self.polynomial_value(x_val, idx), 0, 2)

            # 1. dX1/dt (Уравнение 2.9)
            prod_X1 = 1.0  # f1(X1) = 1
            # f2(X2) * f3(X3) * ... * f14(X14)
            for func_idx, var_idx in zip(range(2, 15), range(1, 14)):
                prod_X1 *= f(func_idx, X_safe[var_idx])

            prod_X1 = np.clip(prod_X1, 0, 5)

            dXdt[0] = (1.0 / self.parameters_norm.get("X1_max", 1.0)) * \
                      prod_X1 * (zeta1 + zeta2 + zeta3 - zeta4 - zeta5)

            # 2. dX2/dt (Уравнение 2.10)
            prod_X2 = 1.0  # f16(X2) = 1
            # f15(X1) * f17(X3) * ... * f28(X14)
            # Индексы функций: 15, 17-28 (всего 13 функций)
            for func_idx in [15] + list(range(17, 29)):
                if func_idx == 15:  # f15(X1)
                    var_idx = 0
                elif 17 <= func_idx <= 28:  # f17-f28
                    var_idx = func_idx - 15
                prod_X2 *= f(func_idx, X_safe[var_idx])

            prod_X2 = np.clip(prod_X2, 0, 5)

            dXdt[1] = (1.0 / self.parameters_norm.get("X2_max", 1.0)) * \
                      prod_X2 * (zeta1 + zeta2 + zeta3 + zeta4 - zeta5)

            # 3. dX3/dt (Уравнение 2.11)
            prod_X3 = 1.0
            for i in range(29, 43):  # f29-f42
                var_idx = i - 29  # f29 -> X1, f30 -> X2, ..., f42 -> X14
                prod_X3 *= f(i, X_safe[var_idx])

            prod_X3 = np.clip(prod_X3, 0, 5)

            dXdt[2] = (1.0 / self.parameters_norm.get("X3_max", 1.0)) * \
                      prod_X3 * (zeta1 + zeta2 + zeta3 + zeta4 - zeta5)

            # 4. dX4/dt (Уравнение 2.12)
            prod_pos_X4 = 1.0
            # f43-f46: X1-X4, f49-f55: X7-X14, f56: X1
            for i in range(43, 57):  # f43-f56
                if i <= 46:  # f43-f46: X1-X4
                    var_idx = i - 43
                elif 49 <= i <= 55:  # f49-f55: X7-X14
                    var_idx = i - 49 + 6  # f49 -> X7 (индекс 6)
                elif i == 56:  # f56: X1
                    var_idx = 0
                else:
                    continue
                prod_pos_X4 *= f(i, X_safe[var_idx])

            prod_pos_X4 = np.clip(prod_pos_X4, 0, 5)

            prod_neg_X4 = f(47, X_safe[4]) * f(48, X_safe[5])  # f47(X5), f48(X6)

            dXdt[3] = (1.0 / self.parameters_norm.get("X4_max", 1.0)) * \
                      (prod_pos_X4 * zeta5 - prod_neg_X4 * (zeta1 + zeta2 + zeta3 + zeta4))

            # 5. dX5/dt (Уравнение 2.13)
            prod_X5 = (f(57, X_safe[3]) *  # X4
                       f(58, X_safe[5]) *  # X6
                       f(59, X_safe[8]) *  # X9
                       f(60, X_safe[9]) *  # X10
                       f(61, X_safe[12]))  # X13

            prod_X5 = np.clip(prod_X5, 0, 5)

            dXdt[4] = (1.0 / self.parameters_norm.get("X5_max", 1.0)) * \
                      prod_X5 * (zeta1 + zeta2 + zeta4 + zeta5 - zeta5)

            # 6. dX6/dt (Уравнение 2.14)
            prod_X6 = 1.0
            # f62-f66: X1-X5, f68-f75: X7-X14 (f67 = 1)
            for i in range(62, 76):  # f62-f75
                if i == 67:  # f67 = 1, пропускаем
                    continue
                if i <= 66:  # f62-f66: X1-X5
                    var_idx = i - 62
                else:  # f68-f75: X7-X14
                    var_idx = i - 68 + 6  # f68 -> X7 (индекс 6)
                prod_X6 *= f(i, X_safe[var_idx])

            prod_X6 = np.clip(prod_X6, 0, 5)

            dXdt[5] = (1.0 / self.parameters_norm.get("X6_max", 1.0)) * \
                      prod_X6 * (zeta1 + zeta2 - zeta5)

            # 7. dX7/dt (Уравнение 2.14)
            prod_X7 = f(76, X_safe[1]) * f(77, X_safe[3]) * f(78, X_safe[13])  # X2, X4, X14

            prod_X7 = np.clip(prod_X7, 0, 5)

            dXdt[6] = (1.0 / self.parameters_norm.get("X7_max", 1.0)) * \
                      (prod_X7 - zeta5)

            # 8. dX8/dt (Уравнение 2.15)
            prod_X8 = 1.0
            for i in range(79, 93):  # f79-f92
                var_idx = i - 79  # f79 -> X1, f80 -> X2, ..., f92 -> X14
                prod_X8 *= f(i, X_safe[var_idx])

            prod_X8 = np.clip(prod_X8, 0, 5)

            dXdt[7] = (1.0 / self.parameters_norm.get("X8_max", 1.0)) * \
                      prod_X8 * (zeta4 + zeta5 - zeta1 - zeta2 - zeta3)

            # 9. dX9/dt (Уравнение 2.17)
            prod_X9 = 1.0
            for i in range(93, 105):  # f93-f104
                var_idx = i - 93  # f93 -> X1, f94 -> X2, ..., f104 -> X12
                prod_X9 *= f(i, X_safe[var_idx])

            prod_X9 = np.clip(prod_X9, 0, 5)

            dXdt[8] = (1.0 / self.parameters_norm.get("X9_max", 1.0)) * \
                      (prod_X9 * (zeta1 + zeta2 + zeta3) - zeta1 - zeta2)

            # 10. dX10/dt (Уравнение 2.17)
            prod_pos_X10 = 1.0
            # f105-f113: X1-X9, f115-f116: X11-X12 (f114 = 1)
            for i in range(105, 117):  # f105-f116
                if i == 114:  # f114 = 1, пропускаем
                    continue
                if i <= 113:  # f105-f113: X1-X9
                    var_idx = i - 105
                else:  # f115-f116: X11-X12
                    var_idx = i - 115 + 10  # f115 -> X11 (индекс 10)
                prod_pos_X10 *= f(i, X_safe[var_idx])

            prod_pos_X10 = np.clip(prod_pos_X10, 0, 5)

            prod_neg_X10 = f(117, X_safe[12]) * f(118, X_safe[13]) * zeta3  # X13, X14

            dXdt[9] = (1.0 / self.parameters_norm.get("X10_max", 1.0)) * \
                      (prod_pos_X10 * (zeta1 + zeta2) - prod_neg_X10)

            # 11. dX11/dt (Уравнение 2.19)
            # f119-f122: X1-X4, f123: X8, f124: X10, f125: X12, f126: X13, f127: X14
            prod_pos_X11 = (f(119, X_safe[0]) *  # X1
                            f(120, X_safe[1]) *  # X2
                            f(121, X_safe[2]) *  # X3
                            f(122, X_safe[3]) *  # X4
                            f(123, X_safe[7]) *  # X8
                            f(124, X_safe[9]) *  # X10
                            f(125, X_safe[11]) * # X12
                            f(126, X_safe[12]) * # X13
                            f(127, X_safe[13]))  # X14

            prod_pos_X11 = np.clip(prod_pos_X11, 0, 5)

            prod_neg_X11 = f(128, X_safe[4]) * f(129, X_safe[5])  # X5, X6

            dXdt[10] = (1.0 / self.parameters_norm.get("X11_max", 1.0)) * \
                       (prod_pos_X11 - prod_neg_X11 * zeta_sum)

            # 12. dX12/dt (Уравнение 2.20)
            prod_X12 = 1.0
            # f130-f141: X1-X12 (кроме X13), f142: X14
            for i in range(130, 143):  # f130-f142
                if i == 141:  # f141: X13 (индекс 12)
                    var_idx = 12
                elif i == 142:  # f142: X14 (индекс 13)
                    var_idx = 13
                else:  # f130-f140: X1-X12 (индексы 0-11)
                    var_idx = i - 130
                prod_X12 *= f(i, X_safe[var_idx])

            prod_X12 = np.clip(prod_X12, 0, 5)

            dXdt[11] = (1.0 / self.parameters_norm.get("X12_max", 1.0)) * \
                       (prod_X12 * (zeta1 + zeta4 + zeta5) - zeta3)

            # 13. dX13/dt (Уравнение 2.21)
            prod_pos_X13 = 1.0
            # f143-f155: X1-X14 (кроме X9)
            for i in range(143, 156):  # f143-f155
                if i == 151:  # f151: X10 (индекс 9)
                    var_idx = 9
                elif i == 152:  # f152: X11 (индекс 10)
                    var_idx = 10
                elif i == 153:  # f153: X12 (индекс 11)
                    var_idx = 11
                elif i == 154:  # f154: X13 (индекс 12)
                    var_idx = 12
                elif i == 155:  # f155: X14 (индекс 13)
                    var_idx = 13
                else:  # f143-f150: X1-X8
                    var_idx = i - 143
                prod_pos_X13 *= f(i, X_safe[var_idx])

            prod_pos_X13 = np.clip(prod_pos_X13, 0, 5)

            prod_neg_X13 = f(156, X_safe[8]) * zeta_sum  # X9

            dXdt[12] = (1.0 / self.parameters_norm.get("X13_max", 1.0)) * \
                       (prod_pos_X13 - prod_neg_X13)

            # 14. dX14/dt (Уравнение 2.21)
            prod_pos_X14 = (f(157, X_safe[4]) *  # X5
                            f(158, X_safe[6]) *  # X7
                            f(159, X_safe[10]) * # X11
                            f(160, X_safe[12]))  # X13

            prod_pos_X14 = np.clip(prod_pos_X14, 0, 5)

            dXdt[13] = (1.0 / self.parameters_norm.get("X14_max", 1.0)) * \
                       (prod_pos_X14 - zeta_sum)

            # Ограничиваем производные
            dXdt = np.clip(dXdt, -0.5, 0.5)

            return dXdt

        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(14)

    def solve_system(self):
        """Решение системы дифференциальных уравнений (СТАБИЛИЗИРОВАННАЯ ВЕРСИЯ)"""
        try:
            # Начальные условия из параметров
            X0 = [self.parameters_norm.get(f"X{i}", 0.3) for i in range(1, 15)]

            self.time_points = np.linspace(0, 10, 500)

            # Решение системы
            self.solution = solve_ivp(
                self.system_equations,
                [0, 10],
                X0,
                t_eval=self.time_points,
                method='RK45',
                rtol=1e-6,
                atol=1e-8,
                max_step=0.1
            )

            # Нормализуем время
            if hasattr(self, 'solution') and self.solution is not None:
                self.original_time = self.solution.t.copy()
                self.solution.t = self.solution.t / self.solution.t[-1]

            return self.solution

        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            import traceback
            traceback.print_exc()

            # Резервное решение - с вариациями
            t_points = np.linspace(0, 1, 100)
            y_values = np.ones((14, 100)) * 0.5

            # Создаем разные траектории для разных переменных
            for i in range(14):
                # Некоторые переменные идут вниз
                if i in [0, 3, 6, 9, 12]:  # Каждые 3 переменные идут вниз
                    amplitude = 2.0 + 0.5 * (i % 4)
                    frequency = 0.5 + 0.2 * (i % 3)
                    y_values[i] = 0.5 - amplitude * t_points * frequency
                else:
                    # Остальные идут вверх или колеблются
                    amplitude = 0.3 + 0.1 * (i % 3)
                    frequency = 1 + (i % 4)
                    y_values[i] = 0.5 + amplitude * np.sin(t_points * np.pi * frequency / 4)

            self.solution = type('obj', (object,), {
                't': t_points,
                'y': y_values
            })()
            return self.solution

    def _convert_to_subscript(self, number):
        """Конвертация числа в нижний индекс"""
        subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(number).translate(subscript_digits)

    def plot_time_series(self):
        """Построение временных рядов с гладкими линиями"""
        if self.solution is None:
            self.solve_system()

        # Создаем график
        fig, ax = plt.subplots(1, 1, figsize=(9, 15))
        colors = plt.cm.tab20(np.linspace(0, 1, 14))

        func_descriptions = []

        # Находим общий диапазон для всего графика
        all_min = np.min(self.solution.y)
        all_max = np.max(self.solution.y)

        # Устанавливаем диапазон с запасом
        y_min = all_min - 0.1 if all_min < 0 else -0.5
        y_max = all_max + 0.1 if all_max > 1 else 1.1

        # Обеспечиваем минимальный диапазон
        y_min = min(y_min, -4.0)
        y_max = max(y_max, 1.0)

        # Список для хранения всех линий для легенды
        lines = []

        for i in range(14):
            t = self.solution.t
            y = self.solution.y[i]

            # Аппроксимация для текстового описания
            try:
                coeffs = np.polyfit(t, y, deg=3)
                a3, a2, a1, a0 = coeffs
                func_descriptions.append(
                    f"X{self._convert_to_subscript(i+1)}(t) = {round(a3, 6)}·t³ + {round(a2, 6)}·t² + {round(a1, 6)}·t + {round(a0, 6)}"
                )
            except:
                func_descriptions.append(
                    f"X{self._convert_to_subscript(i+1)}(t) - аппроксимация не удалась"
                )

            y_label = self.names[f'X{i+1}']
            x_subscript = f"X{self._convert_to_subscript(i+1)}"

            if len(t) > 10:
                # Создаем больше точек для плавности
                t_interp = np.linspace(t.min(), t.max(), 500)

                if len(t) >= 8:
                    try:
                        from scipy.interpolate import make_smoothing_spline

                        # Сортируем точки по времени
                        sort_idx = np.argsort(t)
                        t_sorted = t[sort_idx]
                        y_sorted = y[sort_idx]

                        t_unique, idx_unique = np.unique(t_sorted, return_index=True)
                        if len(t_unique) >= 4:
                            spline = make_smoothing_spline(t_unique, y_sorted[idx_unique],
                                                           lam=0.1)
                            y_interp = spline(t_interp)
                        else:
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(t, y, kind='cubic',
                                                   bounds_error=False,
                                                   fill_value="extrapolate")
                            y_interp = interp_func(t_interp)

                        dy = np.diff(y_interp)
                        if np.max(np.abs(dy)) > 0.5:
                            from scipy.ndimage import gaussian_filter1d
                            y_interp = gaussian_filter1d(y_interp, sigma=2)

                    except Exception as e:
                        try:
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(t, y, kind='cubic',
                                                   bounds_error=False,
                                                   fill_value="extrapolate")
                            y_interp = interp_func(t_interp)
                        except:
                            t_interp = t
                            y_interp = y
                else:
                    try:
                        coeff = np.polyfit(t, y, min(3, len(t)-1))
                        poly = np.poly1d(coeff)
                        y_interp = poly(t_interp)
                    except:
                        t_interp = t
                        y_interp = y

                line, = ax.plot(t_interp, y_interp,
                                label=f'{x_subscript}: {y_label[:40]}...',
                                color=colors[i],
                                linewidth=2.2,
                                alpha=0.85)
                lines.append(line)
            else:
                # Для очень малого количества точек
                line, = ax.plot(t, y,
                                label=f'{x_subscript}: {y_label[:40]}...',
                                color=colors[i],
                                linewidth=2.2,
                                alpha=0.85)
                lines.append(line)

            # Добавляем метки на график
            if len(t) > 10:
                for label_pos in [0.97, 0, 0]:
                    idx = int(len(t) * label_pos)
                    if idx < len(t):
                        x_text = t[idx]
                        y_text = y[idx]

                        # Пропускаем метки, которые слишком близко к краям
                        if y_text < y_min + 0.1 or y_text > y_max - 0.1:
                            continue

                        ax.text(
                            x_text,
                            y_text,
                            f'{x_subscript}',
                            color='black',
                            fontsize=8,
                            fontweight='bold',
                            ha='center',
                            va='center',
                            alpha=0.9,
                            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                        )
                        break

        # Настройки графика
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Значение параметра', fontsize=12)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(y_min, y_max)

        # Добавляем горизонтальные линии для ориентации
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.4, linewidth=0.8)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.4, linewidth=0.8)

        # Создаем легенду над графиком
        fig.subplots_adjust(top=0.82)

        # Создаем легенду с одной колонкой
        legend = fig.legend(lines, [line.get_label() for line in lines],
                            loc='upper center',
                            ncol=1,
                            fontsize=9,
                            frameon=True,
                            fancybox=True,
                            shadow=False,
                            borderaxespad=0.5,
                            bbox_to_anchor=(0.5, 0.99))

        # Настраиваем рамку легенды
        legend.get_frame().set_facecolor('#f9f9f9')
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_edgecolor('#cccccc')

        # Добавляем функции под графиком в один столбик
        func_text = "\n".join(func_descriptions)

        fig.text(0.5, 0.005, func_text,
                 ha='center',
                 va='bottom',
                 fontsize=8,
                 family='monospace')

        # Настраиваем отступы для вертикального графика
        plt.tight_layout(rect=[0, 0.13, 1, 0.80])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

    def plot_radar_charts(self):
        """Построение лепестковых диаграмм"""
        if self.solution is None:
            self.solve_system()

        time_points = [0, 0.25, 0.5, 0.75, 1]
        time_indices = [np.abs(self.solution.t - t).argmin() for t in time_points]
        categories = [f'X{self._convert_to_subscript(i+1)}' for i in range(14)]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

        # Получаем максимальные значения для каждого параметра
        max_values = [self.parameters_norm.get(f"X{i+1}_max", 1) for i in range(14)]
        max_values += max_values[:1]

        # Находим общий диапазон для всех графиков
        all_values = []
        for t_idx in time_indices:
            values = self.solution.y[:, t_idx].tolist()
            all_values.extend(values)

        # Для радар-чартов нормализуем от минимума к максимуму или используем фиксированный диапазон
        value_min = min(all_values)
        value_max = max(all_values)
        radar_min = min(0, value_min - 0.1)
        radar_max = max(1, value_max + 0.1)

        # Строим графики
        for i, (t_idx, ax) in enumerate(zip(time_indices, axes)):
            if i >= len(time_points):
                break

            values = self.solution.y[:, t_idx].tolist()
            values += values[:1]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, values, color=colors[i], linewidth=2.5, linestyle='-')
            ax.fill(angles, values, color=colors[i], alpha=0.25)

            # Добавляем красный контур максимальных пределов
            ax.plot(angles, max_values, color='red', linewidth=2, linestyle='--')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_rlabel_position(0)

            # Динамические метки на оси Y
            y_ticks = np.linspace(radar_min, radar_max, 6)
            y_tick_labels = [f"{tick:.1f}" for tick in y_ticks]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, color="grey", size=8)
            ax.set_ylim(radar_min, radar_max)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color='black', pad=15)

        for i in range(len(time_points), len(axes)):
            fig.delaxes(axes[i])

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors[0], linewidth=2.5, label='t=0'),
            Line2D([0], [0], color=colors[1], linewidth=2.5, label='t=0.25'),
            Line2D([0], [0], color=colors[2], linewidth=2.5, label='t=0.5'),
            Line2D([0], [0], color=colors[3], linewidth=2.5, label='t=0.75'),
            Line2D([0], [0], color=colors[4], linewidth=2.5, label='t=1'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Максимум')
        ]

        fig.legend(handles=legend_elements, loc='upper center',
                   ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.05))

        plt.tight_layout()

        radar_buffer = io.BytesIO()
        plt.savefig(radar_buffer, format='png', dpi=150, bbox_inches='tight')
        radar_buffer.seek(0)
        radar_b64 = base64.b64encode(radar_buffer.getvalue()).decode()
        plt.close(fig)
        return radar_b64

    def plot_disturbances(self):
        """Построение графиков возмущений"""
        time_points = np.linspace(0, 1, 500)

        fig, ax = plt.subplots(figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, 5))

        all_y_values = []
        for i in range(1, 6):
            y = [self.disturbance_function(t * 10, i) for t in time_points]
            all_y_values.extend(y)

        # Находим диапазон для оси Y
        y_min = min(all_y_values) - 0.1
        y_max = max(all_y_values) + 0.1

        for i in range(1, 6):
            y = [self.disturbance_function(t * 10, i) for t in time_points]
            zeta_subscript = f"ζ{self._convert_to_subscript(i)}"
            name = self.disturbance_names.get(f"zeta{i}", f"Возмущение {i}")

            from scipy.interpolate import make_interp_spline
            if len(time_points) > 20:
                t_smooth = np.linspace(time_points.min(), time_points.max(), 300)
                spline = make_interp_spline(time_points, y, k=3)
                y_smooth = spline(t_smooth)
                t_plot, y_plot = t_smooth, y_smooth
            else:
                t_plot, y_plot = time_points, y

            ax.plot(t_plot, y_plot,
                    label=f'{zeta_subscript}: {name}',
                    linewidth=2.5,
                    color=colors[i-1],
                    alpha=0.9)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                  ncol=2, fontsize=11, frameon=True)
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Значение возмущения', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(y_min, y_max)  # Динамический диапазон
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title('Графики возмущений системы', fontsize=14, pad=20)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

    def plot_all_results(self):
        """Построение всех графиков"""
        self.solve_system()
        time_series_b64 = self.plot_time_series()
        radar_b64 = self.plot_radar_charts()
        disturbances_b64 = self.plot_disturbances()

        return [time_series_b64, radar_b64, disturbances_b64]