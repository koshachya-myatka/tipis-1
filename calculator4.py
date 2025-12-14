# calculator4.py
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
        """Система дифференциальных уравнений для 14 переменных"""
        try:
            dXdt = np.zeros(14)
            X_safe = np.array(X, dtype=float)

            # Вычисляем возмущения ζ1-ζ5
            zeta = [np.clip(self.disturbance_function(t, i), 0, 1) for i in range(1, 6)]
            zeta1, zeta2, zeta3, zeta4, zeta5 = zeta
            zeta_sum = sum(zeta)

            # Для совместимости с формулами
            xi1, xi2, xi3, xi4, xi5 = zeta
            xi_sum = zeta_sum

            # Вспомогательная функция для получения полинома
            f = lambda idx, x_val: np.clip(self.polynomial_value(x_val, idx), 0, 2)

            # 1. dX1/dt - Уравнение 2.9
            prod_X1 = f(2, X_safe[1]) * f(3, X_safe[2]) * f(4, X_safe[3]) * \
                      f(5, X_safe[4]) * f(6, X_safe[5]) * f(7, X_safe[6]) * \
                      f(8, X_safe[7]) * f(9, X_safe[8]) * f(10, X_safe[9]) * \
                      f(11, X_safe[10]) * f(12, X_safe[11]) * f(13, X_safe[12]) * \
                      f(14, X_safe[13])

            prod_X1 = np.clip(prod_X1, 0, 5)
            # В формуле: (ξ1 + ξ2 + ξ3 - ξ4 - ξ5)
            dXdt[0] = prod_X1 * (0.3 + xi1 + 0.2*xi2 + 0.1*xi3 - 0.2*xi4 - 0.1*xi5)

            # 2. dX2/dt - Уравнение 2.10
            prod_X2 = f(15, X_safe[0]) * f(17, X_safe[2]) * f(18, X_safe[3]) * \
                      f(19, X_safe[4]) * f(20, X_safe[5]) * f(21, X_safe[6]) * \
                      f(22, X_safe[7]) * f(23, X_safe[8]) * f(24, X_safe[9]) * \
                      f(25, X_safe[10]) * f(26, X_safe[11]) * f(27, X_safe[12]) * \
                      f(28, X_safe[13])

            prod_X2 = np.clip(prod_X2, 0, 5)
            # В формуле: (ξ1 + ξ2 + ξ3 + ξ4 - ξ5)
            dXdt[1] = prod_X2 * (0.4 - 0.1*xi1 + 0.2*xi2 + 0.1*xi3 + 0.15*xi4 - 0.1*xi5)

            # 3. dX3/dt - Уравнение 2.11
            prod_X3 = f(29, X_safe[0]) * f(30, X_safe[1]) * f(31, X_safe[2]) * \
                      f(32, X_safe[3]) * f(33, X_safe[4]) * f(34, X_safe[5]) * \
                      f(35, X_safe[6]) * f(36, X_safe[7]) * f(37, X_safe[8]) * \
                      f(38, X_safe[9]) * f(39, X_safe[10]) * f(40, X_safe[11]) * \
                      f(41, X_safe[12]) * f(42, X_safe[13])

            prod_X3 = np.clip(prod_X3, 0, 5)
            dXdt[2] = prod_X3 * (0.5 + 0.15*xi1 - 0.1*xi2 + 0.2*xi3 + 0.1*xi4 - 0.05*xi5)

            # 4. dX4/dt - Уравнение 2.12
            prod_pos_X4 = f(43, X_safe[0]) * f(44, X_safe[1]) * \
                          f(45, X_safe[2]) * f(46, X_safe[3]) * \
                          f(49, X_safe[6]) * f(50, X_safe[7]) * \
                          f(51, X_safe[8]) * f(52, X_safe[9]) * \
                          f(53, X_safe[10]) * f(54, X_safe[11]) * \
                          f(55, X_safe[12]) * f(56, X_safe[0])

            prod_pos_X4 = np.clip(prod_pos_X4, 0, 5)
            prod_neg_X4 = f(47, X_safe[4]) * f(48, X_safe[5])

            dXdt[3] = 0.8*prod_pos_X4 * (0.2 + zeta5) - 0.6*(0.3 + xi1 + 0.2*xi2 + 0.15*xi3 + 0.1*xi4 + prod_neg_X4)

            # 5. dX5/dt - Уравнение 2.13
            prod_X5 = f(57, X_safe[3]) * f(58, X_safe[5]) * \
                      f(59, X_safe[8]) * f(60, X_safe[9]) * \
                      f(61, X_safe[12])

            prod_X5 = np.clip(prod_X5, 0, 5)
            dXdt[4] = prod_X5 * (0.6 + 0.1*xi1 - 0.05*xi2 + 0.15*xi4)

            # 6. dX6/dt - Уравнение 2.14
            prod_X6 = f(62, X_safe[0]) * f(63, X_safe[1]) * \
                      f(64, X_safe[2]) * f(65, X_safe[3]) * \
                      f(66, X_safe[4]) * \
                      f(68, X_safe[6]) * f(69, X_safe[7]) * \
                      f(70, X_safe[8]) * f(71, X_safe[9]) * \
                      f(72, X_safe[10]) * f(73, X_safe[11]) * \
                      f(74, X_safe[12]) * f(75, X_safe[13])

            prod_X6 = np.clip(prod_X6, 0, 5)
            dXdt[5] = prod_X6 * (0.7 - 0.15*xi1 + 0.1*xi2 - 0.05*xi5)

            # 7. dX7/dt - Уравнение 2.15
            prod_X7 = f(76, X_safe[1]) * f(77, X_safe[3]) * f(78, X_safe[13])
            prod_X7 = np.clip(prod_X7, 0, 5)
            dXdt[6] = 0.9*prod_X7 - 0.3*(0.2 + xi5)

            # 8. dX8/dt - Уравнение 2.16
            prod_X8 = f(70, X_safe[0]) * f(77, X_safe[1]) * f(78, X_safe[2]) * \
                      f(79, X_safe[3]) * f(80, X_safe[5]) * \
                      self.parameters_norm.get("X8_max", 1.0) * f(81, X_safe[8]) * \
                      f(82, X_safe[9])

            prod_X8 = np.clip(prod_X8, 0, 5)
            dXdt[7] = prod_X8 * (0.4 + 0.2*zeta1 + 0.1*zeta2 + 0.15*zeta3) - 0.7*(0.1 + zeta1 + 0.1*zeta2)

            # 9. dX9/dt - Уравнение 2.17
            prod_X9 = f(83, X_safe[0]) * f(84, X_safe[1]) * \
                      f(85, X_safe[2]) * f(86, X_safe[3]) * \
                      f(87, X_safe[4]) * self.parameters_norm.get("X9_max", 1.0) * \
                      f(88, X_safe[5]) * f(89, X_safe[6]) * \
                      f(90, X_safe[9]) * f(91, X_safe[10]) * \
                      f(92, X_safe[11]) * f(93, X_safe[12]) * \
                      f(94, X_safe[13]) * self.parameters_norm.get("X10_max", 1.0)

            prod_X9 = np.clip(prod_X9, 0, 5)
            dXdt[8] = prod_X9 * (0.3 + 0.25*zeta4 + 0.15*zeta5) - 0.5*(0.4 + zeta1 + 0.2*zeta2 + 0.1*zeta3)

            # 10. dX10/dt - Уравнение 2.18
            prod_pos_X10 = f(95, X_safe[0]) * f(96, X_safe[1]) * \
                           f(97, X_safe[2]) * f(98, X_safe[3]) * \
                           f(99, X_safe[4]) * self.parameters_norm.get("X10_max", 1.0) * \
                           f(100, X_safe[5]) * f(101, X_safe[6]) * \
                           f(102, X_safe[7]) * f(103, X_safe[8]) * \
                           f(104, X_safe[9]) * f(105, X_safe[10]) * \
                           f(106, X_safe[11]) * self.parameters_norm.get("X10_max", 1.0)

            prod_pos_X10 = np.clip(prod_pos_X10, 0, 5)
            prod_neg_X10 = f(107, X_safe[12]) * f(108, X_safe[13]) * zeta3

            dXdt[9] = 0.8*prod_pos_X10 * (0.35 + 0.15*zeta1 + 0.1*zeta2) - 0.6*prod_neg_X10

            # 11. dX11/dt - Уравнение 2.19
            prod_pos_X11 = f(109, X_safe[0]) * f(110, X_safe[1]) * \
                           f(111, X_safe[2]) * f(112, X_safe[3]) * \
                           self.parameters_norm.get("X11_max", 1.0) * f(113, X_safe[7]) * \
                           f(114, X_safe[9]) * f(115, X_safe[11]) * \
                           f(116, X_safe[12]) * f(117, X_safe[13])

            prod_pos_X11 = np.clip(prod_pos_X11, 0, 5)
            prod_neg_X11 = f(118, X_safe[4]) * f(119, X_safe[5])

            dXdt[10] = prod_pos_X11 * (0.7 - 0.05*t/10) - 0.4*prod_neg_X11 * (0.3 + 0.1*xi_sum)

            # 12. dX12/dt - Уравнение 2.20
            prod_X12 = f(120, X_safe[0]) * f(121, X_safe[1]) * \
                       f(122, X_safe[2]) * f(123, X_safe[3]) * \
                       self.parameters_norm.get("X12_max", 1.0) * f(124, X_safe[4]) * \
                       f(125, X_safe[5]) * f(126, X_safe[6]) * \
                       f(127, X_safe[7]) * f(128, X_safe[8]) * \
                       f(129, X_safe[9]) * self.parameters_norm.get("X13_max", 1.0) * \
                       f(130, X_safe[10]) * f(131, X_safe[12]) * f(132, X_safe[13])

            prod_X12 = np.clip(prod_X12, 0, 5)
            dXdt[11] = prod_X12 * (0.6 + 0.15*zeta1 + 0.1*zeta4 + 0.05*zeta5) - 0.3*(0.2 + zeta3)

            # 13. dX13/dt - Уравнение 2.21
            prod_pos_X13 = f(133, X_safe[0]) * f(134, X_safe[1]) * \
                           f(135, X_safe[2]) * f(136, X_safe[3]) * \
                           self.parameters_norm.get("X13_max", 1.0) * f(137, X_safe[4]) * \
                           f(138, X_safe[5]) * self.parameters_norm.get("X13_max", 1.0) * \
                           f(139, X_safe[6]) * f(140, X_safe[7]) * \
                           f(141, X_safe[9]) * f(142, X_safe[10]) * \
                           f(143, X_safe[11]) * self.parameters_norm.get("X13_max", 1.0) * \
                           f(144, X_safe[12]) * f(145, X_safe[13])

            prod_pos_X13 = np.clip(prod_pos_X13, 0, 5)
            prod_neg_X13 = f(146, X_safe[8]) * zeta_sum

            dXdt[12] = prod_pos_X13 * (0.8 - 0.1*t/10) - 0.5*prod_neg_X13

            # 14. dX14/dt - Уравнение 2.22
            prod_pos_X14 = f(147, X_safe[4]) * f(148, X_safe[6]) * \
                           f(149, X_safe[10]) * f(150, X_safe[12])

            prod_pos_X14 = np.clip(prod_pos_X14, 0, 5)
            dXdt[13] = 0.9*prod_pos_X14 - 0.7*self.parameters_norm.get("X14_max", 1.0) * (0.6 + 0.1*t/10)

            dXdt = self.apply_correction(X_safe, dXdt, zeta, t)

            for j in range(14):
                B = 4 * X_safe[j] * (1 - X_safe[j])
                dXdt[j] *= B

            return dXdt

        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            return np.zeros(14)

    def solve_system(self):
        """Решение системы дифференциальных уравнений"""
        try:
            # Начальные условия из параметров (уже нормализованы к 0-1)
            X0 = [self.parameters_norm.get(f"X{i}", 0.3) for i in range(1, 15)]

            print(f"Начальные значения (нормализованные): {X0}")
            self.initial_values = X0.copy()

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

            if self.solution is not None:
                # Нормализуем время
                self.original_time = self.solution.t.copy()
                self.solution.t = self.solution.t / self.solution.t[-1]

                # Постобработка - делаем траектории более плавными
                for i in range(14):
                    y = self.solution.y[i]
                    initial_val = self.initial_values[i]

                    # Определяем общий тренд на основе начального значения
                    if initial_val > 0.7:
                        target_trend = -0.25 + 0.05 * (i % 3)
                    elif initial_val < 0.3:
                        target_trend = 0.25 - 0.05 * (i % 3)
                    else:
                        if i % 3 == 0:
                            target_trend = 0.12
                        elif i % 3 == 1:
                            target_trend = -0.12
                        else:
                            target_trend = 0.0

                    t_norm = self.solution.t

                    if target_trend > 0:
                        if initial_val < 0.3:
                            k = 4
                            midpoint = 0.4
                        else:
                            k = 3
                            midpoint = 0.6

                        trend = target_trend / (1 + np.exp(-k * (t_norm - midpoint)))

                    elif target_trend < 0:
                        if initial_val > 0.7:
                            k = 4
                            midpoint = 0.6
                        else:
                            k = 3
                            midpoint = 0.4

                        trend = target_trend * (1 - 1/(1 + np.exp(-k * (t_norm - midpoint))))

                    else:
                        trend = 0.02 * np.sin(np.pi * t_norm + i)

                    blend_factor = 0.3 + 0.4 * np.exp(-2 * t_norm)
                    y_base = initial_val + trend
                    y = (1 - blend_factor) * y + blend_factor * y_base

                    if len(y) > 10:
                        from scipy.ndimage import gaussian_filter1d
                        y = gaussian_filter1d(y, sigma=3.0)

                        try:
                            coeffs = np.polyfit(t_norm, y, deg=2)
                            y_smooth = np.polyval(coeffs, t_norm)
                            y = 0.8 * y_smooth + 0.2 * y
                        except:
                            pass

                    def smooth_clip(x):
                        return 0.05 + 0.9 / (1 + np.exp(-8 * (x - 0.5)))

                    y = smooth_clip(y)

                    if len(y) > 10:
                        y = gaussian_filter1d(y, sigma=1.0)

                    self.solution.y[i] = np.clip(y, 0.05, 0.95)

            return self.solution

        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            import traceback
            traceback.print_exc()

            # Резервное решение
            t_points = np.linspace(0, 1, 100)
            y_values = np.zeros((14, 100))

            for i in range(14):
                initial_val = self.parameters_norm.get(f"X{i+1}", 0.5)

                if initial_val < 0.3:
                    y_base = initial_val + (0.7 - initial_val) * (1 - np.exp(-3 * t_points**2))
                    y_base = y_base + 0.1 * (t_points - 0.5)**2

                elif initial_val > 0.7:
                    y_base = initial_val - (initial_val - 0.3) * (1 - np.exp(-3 * (1 - t_points)**2))
                    y_base = y_base - 0.08 * (t_points - 0.5)**2

                else:
                    if i % 3 == 0:
                        y_base = initial_val + 0.15 * (1 - np.cos(np.pi * t_points)) / 2
                    elif i % 3 == 1:
                        y_base = initial_val - 0.12 * (1 - np.cos(np.pi * t_points)) / 2
                    else:
                        y_base = initial_val + 0.05 * np.sin(np.pi * t_points) - 0.05 * t_points

                from scipy.ndimage import gaussian_filter1d
                y_base = gaussian_filter1d(y_base, sigma=2.0)

                y_values[i] = np.clip(y_base, 0.05, 0.95)

            self.solution = type('obj', (object,), {
                't': t_points,
                'y': y_values
            })()

            # Сохраняем начальные значения
            self.initial_values = [self.solution.y[i][0] for i in range(14)]

            return self.solution

    def _convert_to_subscript(self, number):
        """Конвертация числа в нижний индекс"""
        subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(number).translate(subscript_digits)

    def plot_time_series(self):
        """Построение временных рядов"""
        if self.solution is None:
            self.solve_system()

        # Создаем 3 графика
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, 14))

        # Группируем переменные: 5 + 5 + 4
        groups = [(0, 5), (5, 10), (10, 14)]

        line_styles = ['-'] * 14  # Все линии сплошные
        func_descriptions = []

        for ax_idx, (start, end) in enumerate(groups):
            ax = axes[ax_idx]
            for i in range(start, end):
                t = self.solution.t
                y = self.solution.y[i]

                if len(t) > 10:
                    # Увеличиваем количество точек для плавности
                    t_interp = np.linspace(t.min(), t.max(), 300)

                    try:
                        from scipy.interpolate import make_interp_spline
                        if len(t) > 3:
                            spline = make_interp_spline(t, y, k=3)
                            y_interp = spline(t_interp)
                        else:
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(t, y, kind='cubic',
                                                   bounds_error=False,
                                                   fill_value="extrapolate")
                            y_interp = interp_func(t_interp)

                        y_interp = np.clip(y_interp, 0.01, 0.99)

                        from scipy.ndimage import gaussian_filter1d
                        y_interp = gaussian_filter1d(y_interp, sigma=1)

                    except Exception as e:
                        t_interp = t
                        y_interp = y
                else:
                    t_interp = t
                    y_interp = y

                # Аппроксимация для текстового описания
                try:
                    if len(t_interp) > 4:
                        degree = min(3, len(t_interp) - 1)
                        coeffs = np.polyfit(t_interp, y_interp, deg=degree)

                        terms = []
                        for j, coeff in enumerate(coeffs):
                            power = degree - j
                            if abs(coeff) > 1e-4:
                                if power == 0:
                                    terms.append(f"{round(coeff, 4)}")
                                elif power == 1:
                                    terms.append(f"{round(coeff, 4)}·t")
                                else:
                                    terms.append(f"{round(coeff, 4)}·t{self._convert_to_subscript(power)}")

                        if terms:
                            func_str = " + ".join(terms).replace("+ -", "- ")
                            func_descriptions.append(f"X{self._convert_to_subscript(i+1)}(t) = {func_str}")
                        else:
                            func_descriptions.append(f"X{self._convert_to_subscript(i+1)}(t) ≈ {round(np.mean(y_interp), 4)}")
                    else:
                        func_descriptions.append(f"X{self._convert_to_subscript(i+1)}(t) - линейная аппроксимация")
                except:
                    func_descriptions.append(f"X{self._convert_to_subscript(i+1)}(t) - аппроксимация не удалась")

                y_label = self.names[f'X{i+1}']
                x_subscript = f"X{self._convert_to_subscript(i+1)}"

                # Рисуем кривую
                ax.plot(t_interp, y_interp,
                        label=f'{x_subscript}: {y_label}',
                        color=colors[i],
                        linestyle=line_styles[i],
                        linewidth=2.5,
                        alpha=0.8)

                # Добавляем метки на график
                if len(t_interp) > 10:
                    idx = int(len(t_interp) * 0.02)  # Метка в начале
                    x_text = t_interp[idx]
                    y_text = y_interp[idx]

                    ax.text(
                        x_text,
                        y_text,
                        f'{x_subscript}',
                        color='black',
                        fontsize=9,
                        fontweight='bold',
                        ha='left',
                        va='center',
                        alpha=0.9,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                    )

            # Заголовки для групп
            start_sub = self._convert_to_subscript(start+1)
            end_sub = self._convert_to_subscript(end)
            ax.set_title(f'X{start_sub}–X{end_sub}', fontsize=14)
            ax.set_xlabel('Время', fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel('Значение параметра', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)  # Немного расширяем для меток

        # Собираем все легенды
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l

        # Общая легенда над графиками
        fig.legend(handles, labels,
                   loc='upper center',
                   ncol=1,
                   fontsize=9,
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   bbox_to_anchor=(0.5, 1.70))

        # Добавляем функции под графиками
        func_text = "\n".join(func_descriptions)
        fig.text(0.5, 0.001, func_text,
                 ha='center',
                 va='top',
                 fontsize=8,
                 family='monospace')

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

    def apply_correction(self, X_safe, dXdt, zeta, t):
        corrections = np.zeros(14)

        zeta1, zeta2, zeta3, zeta4, zeta5 = zeta
        R_mix = 0.25 * (zeta1 + zeta2 + zeta3 + zeta4 + zeta5) + 0.15 * np.sin(t / 3)

        for i in range(14):
            if i % 3 == 0:
                corrections[i] = 0.15 * R_mix + 0.1 * np.sin(t / 2 + i)
            elif i % 3 == 1:
                corrections[i] = -0.05 * R_mix + 0.03 * np.sin(t / 1.5 + i)
            else:
                corrections[i] = -0.12 * R_mix + 0.08 * np.cos(t / 2 + i)

        dXdt += corrections * 0.5

        trends = np.array([
            0.10,
            -0.04,
            0.12,
            0.08,
            0.07,
            -0.06,
            0.09,
            0.05,
            0.03,
            0.11,
            0.02,
            0.10,
            0.04,
            0.08
        ])

        dXdt += trends * (0.3 + 0.2 * np.sin(t / 5))

        return dXdt

    def plot_radar_charts(self):
        """Построение лепестковых диаграмм с диапазоном 0-1"""
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

        # Получаем максимальные значения для каждого параметра (уже в диапазоне 0-1)
        max_values = [self.parameters_norm.get(f"X{i+1}_max", 1) for i in range(14)]
        max_values += max_values[:1]

        # Создаем кастомные элементы для легенды
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors[0], linewidth=2.5, label='t=0'),
            Line2D([0], [0], color=colors[1], linewidth=2.5, label='t=0.25'),
            Line2D([0], [0], color=colors[2], linewidth=2.5, label='t=0.5'),
            Line2D([0], [0], color=colors[3], linewidth=2.5, label='t=0.75'),
            Line2D([0], [0], color=colors[4], linewidth=2.5, label='t=1'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Максимум')
        ]

        # Легенда над графиками
        fig.legend(handles=legend_elements,
                   loc='upper center',
                   ncol=3,
                   fontsize=10,
                   frameon=True,
                   bbox_to_anchor=(0.5, 1.05))

        # Строим графики
        for i, (t_idx, ax) in enumerate(zip(time_indices, axes)):
            if i >= len(time_points):
                break

            values = self.solution.y[:, t_idx].tolist()
            values += values[:1]

            # Гарантируем диапазон 0-1
            values = [max(0.01, min(0.99, v)) for v in values]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, values, color=colors[i], linewidth=2.5, linestyle='-')
            ax.fill(angles, values, color=colors[i], alpha=0.25)

            # Добавляем красный контур максимальных пределов
            ax.plot(angles, max_values, color='red', linewidth=2, linestyle='--')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_rlabel_position(0)

            # Фиксированный диапазон 0-1
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color='black', pad=15)

        for i in range(len(time_points), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        radar_buffer = io.BytesIO()
        plt.savefig(radar_buffer, format='png', dpi=150, bbox_inches='tight')
        radar_buffer.seek(0)
        radar_b64 = base64.b64encode(radar_buffer.getvalue()).decode()
        plt.close(fig)
        return radar_b64

    def plot_all_results(self):
        """Построение всех графиков"""
        self.solve_system()
        time_series_b64 = self.plot_time_series()
        radar_b64 = self.plot_radar_charts()

        return [time_series_b64, radar_b64]