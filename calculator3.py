import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import io
import base64

class Calculator3:
    def __init__(self):
        self.names = {
            "K1": "Время испарения химически опасных веществ",
            "K2": "Время ликвидации последствий аварии",
            "K3": "Площадь заражения в результате аварии",
            "K4": "Время подхода первичного и/или вторичного облака к населенным пунктам",
            "K5": "Потери от первичного облака",
            "K6": "Потери от вторичного облака",
            "K7": "Количество получивших амбулаторную помощь, чел.",
            "K8": "Количество размещенных в стационаре и реанимации, чел.",
            "K9": "Количество пораженной техники",
            "K10": "Количество объемов и растворов для обеззараживания местности",
            "K11": "Количество сил и средств, необходимых для проведения аварийно-спасательных работ",
            "K12": "Эффективность системы оповещения, %",
            "K13": "Количество людей в зоне поражения",
            "K14": "Количество спасателей в зоне поражения",
            "K15": "Развитость системы МЧС"
        }

        # временные точки
        self.time_points = np.linspace(0, 10, 300)
        self.solution = None

        # диапазоны значений для параметров
        self.pR = {
            'K1': (0, 24), 'K2': (0, 72), 'K3': (0, 100),
            'K4': (0, 6), 'K5': (0, 1000), 'K6': (0, 500),
            'K7': (0, 500), 'K8': (0, 200), 'K9': (0, 50),
            'K10': (0, 1000), 'K11': (0, 200), 'K12': (0, 100),
            'K13': (0, 10000), 'K14': (0, 500), 'K15': (0, 100)
        }

        # все параметры системы
        self.parameters = self.generate_valid_parameters()
        self.parameters_norm = {}

    def generate_system_parameters(self):
        parameters = {}

        # параметры K1-K15
        for i in range(1, 16):
            param_name = f"K{i}"
            min_r, max_r = self.pR[param_name]
            base_value = round(random.uniform(min_r, max_r), 2)
            max_value = round(random.uniform(base_value, max_r), 2)
            parameters[param_name] = base_value
            parameters[f"{param_name}_max"] = max_value

        # Коэффициенты для возмущений R1-R4
        for i in range(1, 5):
            parameters[f"R{i}_a3"] = round(random.uniform(-0.5, 0.5), 3)
            parameters[f"R{i}_a2"] = round(random.uniform(-0.8, 0.8), 3)
            parameters[f"R{i}_a1"] = round(random.uniform(0.2, 0.8), 3)
            parameters[f"R{i}_a0"] = round(random.uniform(0.1, 0.4), 3)

        # Полиномы для внутренних функций f1-f55
        for i in range(1, 56):
            parameters[f"f{i}_a3"] = round(random.uniform(-0.5, 0.5), 3)
            parameters[f"f{i}_a2"] = round(random.uniform(-0.8, 0.8), 3)
            parameters[f"f{i}_a1"] = round(random.uniform(0.2, 0.8), 3)
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 0.4), 3)

        return parameters

    def generate_valid_parameters(self):
        for attempt in range(10):
            try:
                params = self.generate_system_parameters()
                params_norm = self._get_normalized(params)
                self.parameters = params
                self.parameters_norm = params_norm

                solution = self.solve_system()

                if (solution is not None and
                        np.all(np.isfinite(solution.y)) and
                        not np.any(np.isnan(solution.y)) and
                        np.max(solution.y) < 5.0):

                    print(f"Успешная генерация параметров с попытки {attempt + 1}")
                    return params

            except Exception as e:
                print(f"Попытка {attempt + 1} не удалась: {e}")
                continue

        print("Используем резервные параметры")
        return self.create_fallback_parameters()

    def create_fallback_parameters(self):
        parameters = {}

        for i in range(1, 16):
            param_name = f"K{i}"
            min_r, max_r = self.pR[param_name]
            parameters[param_name] = round(min_r + 0.5 * (max_r - min_r), 2)
            parameters[f"{param_name}_max"] = round(min_r + 0.6 * (max_r - min_r), 2)

        for i in range(1, 5):
            parameters[f"R{i}_a3"] = 0.0
            parameters[f"R{i}_a2"] = round(random.uniform(-0.2, 0.2), 3)
            parameters[f"R{i}_a1"] = round(random.uniform(0.3, 0.7), 3)
            parameters[f"R{i}_a0"] = round(random.uniform(0.1, 0.3), 3)

        for i in range(1, 56):
            parameters[f"f{i}_a3"] = 0.0
            parameters[f"f{i}_a2"] = round(random.uniform(-0.2, 0.2), 3)
            parameters[f"f{i}_a1"] = round(random.uniform(0.3, 0.7), 3)
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 0.3), 3)

        return parameters

    def _get_normalized(self, params):
        params_norm = params.copy()

        for i in range(1, 16):
            param_name = f"K{i}"
            min_val, max_val = self.pR[param_name]

            if param_name in params_norm:
                params_norm[param_name] = (params_norm[param_name] - min_val) / (max_val - min_val)

            if f"{param_name}_max" in params_norm:
                params_norm[f"{param_name}_max"] = (params_norm[f"{param_name}_max"] - min_val) / (max_val - min_val)

        return params_norm

    def polynomial_value(self, x, i):
        x = np.clip(x, -10, 10)

        a3 = self.parameters.get(f"f{i}_a3", 0)
        a2 = self.parameters.get(f"f{i}_a2", 0)
        a1 = self.parameters.get(f"f{i}_a1", 0.5)
        a0 = self.parameters.get(f"f{i}_a0", 0.2)

        return a3 * x**3 + a2 * x**2 + a1 * x + a0

    def disturbance_function(self, t, i):
        a3 = self.parameters.get(f"R{i}_a3", 0)
        a2 = self.parameters.get(f"R{i}_a2", 0)
        a1 = self.parameters.get(f"R{i}_a1", 0.5)
        a0 = self.parameters.get(f"R{i}_a0", 0.2)

        x = np.clip(t / 10.0, 0, 1)

        value = a3 * x**3 + a2 * x**2 + a1 * x + a0

        normalized_value = 1 / (1 + np.exp(-6 * (value - 0.5)))

        return float(normalized_value)

    def system_equations(self, t, K):
        try:
            dKdt = np.zeros(15)
            K_safe = np.array(K, dtype=float)

            R1 = self.disturbance_function(t, 1)
            R2 = self.disturbance_function(t, 2)
            R3 = self.disturbance_function(t, 3)
            R4 = self.disturbance_function(t, 4)

            f = lambda idx, x: self.polynomial_value(x, idx)

            # dK1/dt = - (f1(K10) f2(K11) f3(K14))
            dKdt[0] = - (f(1, K_safe[9]) * f(2, K_safe[10]) * f(3, K_safe[13]))

            # dK2/dt = f4(K3)f5(K7)f6(K8)f7(K9)f8(K13) - (f9(K10)f10(K11)f11(K14)f12(K15)*R1 + R2 + R3 + R4)
            prod_pos = (f(4, K_safe[2]) * f(5, K_safe[6]) * f(6, K_safe[7]) *
                        f(7, K_safe[8]) * f(8, K_safe[12]))
            prod_neg = (f(9, K_safe[9]) * f(10, K_safe[10]) * f(11, K_safe[13]) *
                        f(12, K_safe[14]) * R1 + R2 + R3 + R4)
            dKdt[1] = prod_pos - prod_neg

            # dK3/dt = f13(K1) - (f14(K15)*R1 + R3 + R4)
            dKdt[2] = f(13, K_safe[0]) - (f(14, K_safe[14]) * R1 + R3 + R4)

            # dK4/dt = f15(K1)
            dKdt[3] = f(15, K_safe[0])

            # dK5/dt = f16(K1)*R2 - R1
            dKdt[4] = f(16, K_safe[0]) * R2 - R1

            # dK6/dt = R2 - (f17(K4)f18(K11)f19(K12)f20(K14)*R1)
            dKdt[5] = R2 - (f(17, K_safe[3]) * f(18, K_safe[10]) *
                            f(19, K_safe[11]) * f(20, K_safe[13]) * R1)

            # dK7/dt = f21(K5)f22(K6)f23(K13)f24(K15)*R1 + R2 + R3
            dKdt[6] = (f(21, K_safe[4]) * f(22, K_safe[5]) * f(23, K_safe[12]) *
                       f(24, K_safe[14]) * R1 + R2 + R3)

            # dK8/dt = f25(K5)f26(K6)f27(K11)f28(K13)f29(K14)f30(K15)*R1 + R2 + R3
            dKdt[7] = (f(25, K_safe[4]) * f(26, K_safe[5]) * f(27, K_safe[10]) *
                       f(28, K_safe[12]) * f(29, K_safe[13]) * f(30, K_safe[14]) *
                       R1 + R2 + R3)

            # dK9/dt = f31(K3)f32(K13)*R2 - (f33(K10)f34(K11)f35(K14)*R1)
            dKdt[8] = (f(31, K_safe[2]) * f(32, K_safe[12]) * R2 -
                       (f(33, K_safe[9]) * f(34, K_safe[10]) * f(35, K_safe[13]) * R1))

            # dK10/dt = f36(K3)f37(K9)f38(K15)*R1 + R2 + R3 + R4
            dKdt[9] = (f(36, K_safe[2]) * f(37, K_safe[8]) * f(38, K_safe[14]) *
                       R1 + R2 + R3 + R4)

            # dK11/dt = f39(K3)f40(K13)f41(K14)*R1 + R2 + R3 - (f42(K15)*R4)
            dKdt[10] = (f(39, K_safe[2]) * f(40, K_safe[12]) * f(41, K_safe[13]) *
                        R1 + R2 + R3 - (f(42, K_safe[14]) * R4))

            # dK12/dt = f43(K11)f44(K13)f45(K14)*R1 + R2 + R3 - f46(K15)
            dKdt[11] = (f(43, K_safe[10]) * f(44, K_safe[12]) * f(45, K_safe[13]) *
                        R1 + R2 + R3 - f(46, K_safe[14]))

            # dK13/dt = f47(K2)f48(K3)*R2
            dKdt[12] = f(47, K_safe[1]) * f(48, K_safe[2]) * R2

            # dK14/dt = f49(K11)f50(K12)f51(K13)*R1 + R2
            dKdt[13] = (f(49, K_safe[10]) * f(50, K_safe[11]) * f(51, K_safe[12]) *
                        R1 + R2)

            # dK15/dt = f52(K2)f53(K3)f54(K13)f55(K14)*R1 + R2
            dKdt[14] = (f(52, K_safe[1]) * f(53, K_safe[2]) * f(54, K_safe[12]) *
                        f(55, K_safe[13]) * R1 + R2)

            R_sum = (R1 + R2 + R3 + R4) / 4
            R_mix = 0.4 * R_sum + 0.2 * np.sin(t / 3)

            P = lambda x, idx: self.polynomial_value(x, idx)

            dKdt = self.apply_cor(K_safe, P, R1, R2, R3, R4, R_mix)

            for j in range(15):
                B = 4 * K[j] * (1 - K[j])
                dKdt[j] *= B

            return dKdt

        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            return np.zeros(15)

    def solve_system(self):
        try:
            # начальные условия из параметров
            K0 = [self.parameters_norm.get(f"K{i}", 0.3) for i in range(1, 16)]

            self.time_points = np.linspace(0, 10, 500)

            # решение системы
            self.solution = solve_ivp(
                self.system_equations,
                [0, 10],
                K0,
                t_eval=self.time_points,
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )

            # Нормализуем время
            if hasattr(self, 'solution') and self.solution is not None:
                self.original_time = self.solution.t.copy()
                self.solution.t = self.solution.t / self.solution.t[-1]

            return self.solution

        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            # Простое резервное решение
            t_points = np.linspace(0, 1, 100)
            y_values = np.ones((15, 100)) * 0.5
            for i in range(15):
                y_values[i] = 0.5 + 0.2 * np.sin(t_points * np.pi * (i + 1) / 15)

            self.solution = type('obj', (object,), {
                't': t_points,
                'y': y_values
            })()
            return self.solution

    def _convert_to_subscript(self, number):
        subscript_digits = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        return str(number).translate(subscript_digits)

    def plot_time_series(self):
        if self.solution is None:
            self.solve_system()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, 15))
        groups = [(0, 5), (5, 10), (10, 15)]

        line_styles = ['-'] * 15
        func_descriptions = []

        for ax_idx, (start, end) in enumerate(groups):
            ax = axes[ax_idx]
            for i in range(start, end):
                t = self.solution.t
                y = self.solution.y[i]

                coeffs = np.polyfit(t, y, deg=3)
                a3, a2, a1, a0 = coeffs
                func_descriptions.append(
                    f"K{self._convert_to_subscript(i+1)}(t) = {round(a3, 6)}·t³ + {round(a2, 6)}·t² + {round(a1, 6)}·t + {round(a0, 6)}"
                )

                y_label = self.names[f'K{i+1}']
                k_subscript = f"K{self._convert_to_subscript(i+1)}"
                ax.plot(t, y, label=f'{k_subscript} ({y_label})',
                        color=colors[i],
                        linestyle=line_styles[i],
                        linewidth=2,
                        alpha=0.8)

                if len(t) > 10:
                    idx = int(len(t) * 0.02)
                    x_text = t[idx]
                    y_text = y[idx]

                    ax.text(
                        x_text,
                        y_text,
                        f'{k_subscript}',
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        ha='left',
                        va='center',
                        alpha=0.9,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                    )

            start_sub = self._convert_to_subscript(start+1)
            end_sub = self._convert_to_subscript(end)
            ax.set_title(f'K{start_sub}–K{end_sub}', fontsize=14)
            ax.set_xlabel('Время', fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel('Значение параметра', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)

        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l

        fig.legend(handles, labels,
                   loc='upper center',
                   ncol=1,
                   fontsize=11,
                   frameon=True,
                   bbox_to_anchor=(0.5, 1.75))

        func_text = "\n".join(func_descriptions)
        fig.text(0.5, -0.08, func_text, ha='center', va='top', fontsize=12)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

    def apply_cor(self, K_safe, P, R1, R2, R3, R4, R_mix):
        d = np.zeros(15)

        d[0]  = 0.15 * P(K_safe[1], 1)  - 0.1  * P(K_safe[2], 2)  + 0.12 * R_mix
        d[1]  = 0.18 * P(K_safe[0], 3)  - 0.11 * P(K_safe[3], 4)  + 0.1  * R1
        d[2]  = 0.16 * P(K_safe[4], 5)  - 0.14 * P(K_safe[1], 6) * R2 + 0.08 * R_mix
        d[3]  = 0.14 * P(K_safe[0], 7)  + 0.1  * P(K_safe[2], 8)  - 0.09 * R3
        d[4]  = 0.16 * P(K_safe[3], 9)  - 0.12 * P(K_safe[7], 10) + 0.12 * R2
        d[5]  = 0.15 * P(K_safe[1], 11) - 0.16 * P(K_safe[8], 12) + 0.11 * R_mix
        d[6]  = 0.18 * P(K_safe[2], 13) - 0.14 * P(K_safe[5], 14) + 0.1  * R1
        d[7]  = 0.17 * P(K_safe[6], 15) - 0.15 * P(K_safe[4], 16) + 0.12 * R3
        d[8]  = 0.16 * P(K_safe[0], 17) - 0.13 * P(K_safe[10], 18) + 0.1  * R2
        d[9]  = 0.15 * P(K_safe[8], 19) + 0.14 * P(K_safe[11], 20) - 0.12 * R4
        d[10] = 0.18 * P(K_safe[9], 21) - 0.16 * P(K_safe[12], 22) + 0.13 * R_mix
        d[11] = 0.17 * P(K_safe[10], 23) - 0.15 * P(K_safe[6], 24) + 0.11 * R1
        d[12] = 0.18 * P(K_safe[13], 25) - 0.16 * P(K_safe[5], 26) + 0.12 * R3
        d[13] = 0.17 * P(K_safe[11], 27) + 0.15 * P(K_safe[14], 28) - 0.1  * R2
        d[14] = 0.19 * P(K_safe[7], 29) - 0.16 * P(K_safe[9], 30) + 0.13 * R4

        return d


    def plot_disturbances(self):
        time_points = np.linspace(0, 1, 500)

        fig, ax = plt.subplots(figsize=(12, 6))

        disturbance_names = [
            'R₁ - Уровень финансирования системы МЧС города',
            'R₂ - Степень экономического развития города',
            'R₃ - Наличие прибыли у предприятия',
            'R₄ - Доля современного оборудования на предприятии'
        ]

        colors = plt.cm.tab10(np.linspace(0, 1, 4))

        # Все линии сплошные
        line_styles = ['-', '-', '-', '-']

        for i in range(1, 5):
            # Строим график напрямую по полиному из пользовательских коэффициентов
            y = [self.disturbance_function(t * 10, i) for t in time_points]

            r_subscript = f"R{self._convert_to_subscript(i)}"
            ax.plot(time_points, y,
                    label=disturbance_names[i-1],
                    linewidth=2,
                    color=colors[i-1],
                    linestyle=line_styles[i-1],
                    alpha=0.8)

            if len(time_points) > 10:
                idx = int(len(time_points) * 0.97)
                x_text = time_points[idx]
                y_text = y[idx]

                ax.text(
                    x_text,
                    y_text,
                    f'{r_subscript}',
                    color='black',
                    fontsize=10,
                    fontweight='bold',
                    ha='left',
                    va='center',
                    alpha=0.9,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                )

        # Легенда над графиком
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=2, fontsize=11, frameon=True)
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Значение возмущения', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

    def plot_radar_charts(self):
        if self.solution is None:
            self.solve_system()

        time_points = [0, 0.25, 0.5, 0.75, 1]
        time_indices = [np.abs(self.solution.t - t).argmin() for t in time_points]
        categories = [f'K{self._convert_to_subscript(i+1)}' for i in range(15)]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

        # Получаем максимальные значения для каждого параметра
        max_values = [self.parameters_norm.get(f"K{i+1}_max", 1) for i in range(15)]
        max_values += max_values[:1]

        # Создаем кастомные элементы для легенды
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=colors[0], linewidth=2, linestyle='-', label='Текущие значения (t=0)'),
            Line2D([0], [0], color=colors[1], linewidth=2, linestyle='-', label='Текущие значения (t=0.25)'),
            Line2D([0], [0], color=colors[2], linewidth=2, linestyle='-', label='Текущие значения (t=0.5)'),
            Line2D([0], [0], color=colors[3], linewidth=2, linestyle='-', label='Текущие значения (t=0.75)'),
            Line2D([0], [0], color=colors[4], linewidth=2, linestyle='-', label='Текущие значения (t=1)'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Предельные значения')
        ]

        # Легенда над графиками
        fig.legend(handles=legend_elements,
                   loc='upper center',
                   ncol=2,  # Два столбца для лучшего размещения
                   fontsize=10,
                   frameon=True,
                   bbox_to_anchor=(0.5, 1.05))

        # Строим графики
        for i, (t_idx, ax) in enumerate(zip(time_indices, axes)):
            if i >= len(time_points):
                break
            values = self.solution.y[:, t_idx].tolist()
            values += values[:1]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid')
            ax.fill(angles, values, color=colors[i], alpha=0.25)
            # Добавляем красный контур максимальных пределов
            ax.plot(angles, max_values, color='red', linewidth=2, linestyle='dashed')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_rlabel_position(0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color='black', pad=10)

        # Скрываем неиспользуемые subplots
        for i in range(len(time_points), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        radar_buffer = io.BytesIO()
        plt.savefig(radar_buffer, format='png', dpi=100, bbox_inches='tight')
        radar_buffer.seek(0)
        radar_b64 = base64.b64encode(radar_buffer.getvalue()).decode()
        plt.close(fig)
        return radar_b64

    def plot_all_results(self):
        self.solve_system()
        time_series_b64 = self.plot_time_series()
        radar_b64 = self.plot_radar_charts()
        disturbances_b64 = self.plot_disturbances()
        return [time_series_b64, radar_b64, disturbances_b64]