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

        # диапазоны значений для параметров (только максимумы)
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

        # параметры K1-K15 (только базовые значения и максимумы)
        for i in range(1, 16):
            param_name = f"K{i}"
            min_r, max_r = self.pR[param_name]
            # Генерируем случайное начальное значение - ближе к минимуму для стабильности
            base_value = round(random.uniform(min_r + 0.1*(max_r-min_r), min_r + 0.3*(max_r-min_r)), 2)
            # Генерируем случайный максимум
            max_value = round(random.uniform(min_r + 0.5*(max_r-min_r), max_r), 2)
            parameters[param_name] = base_value
            parameters[f"{param_name}_max"] = max_value

        # ВМЕСТО коэффициентов задаем прямые значения R1-R4
        for i in range(1, 5):
            parameters[f"R{i}"] = round(random.uniform(0.4, 0.7), 2)  # Более узкий диапазон

        # полиномы для внутренних функций f1-f55 - умеренные коэффициенты для стабильности
        for i in range(1, 56):
            # Умеренные коэффициенты для стабильности, но достаточные для кривизны
            parameters[f"f{i}_a3"] = round(random.uniform(-0.1, 0.1), 4)
            parameters[f"f{i}_a2"] = round(random.uniform(-0.2, 0.2), 4)
            parameters[f"f{i}_a1"] = round(random.uniform(0.3, 0.8), 4)  # Положительный для роста
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 0.4), 4)   # Положительный для стабильности

        return parameters

    def generate_valid_parameters(self):
        # Увеличиваем количество попыток
        for attempt in range(20):  # Увеличил до 20 попыток
            try:
                params = self.generate_system_parameters()
                params_norm = self._get_normalized(params)
                self.parameters = params
                self.parameters_norm = params_norm

                # Пробуем решить систему
                self.solve_system()

                # Проверяем, что все значения неотрицательные и система стабильна
                if (self.solution is not None and
                        np.all(self.solution.y >= -1e-6) and
                        np.all(np.isfinite(self.solution.y)) and
                        not np.any(np.isnan(self.solution.y))):

                    # Дополнительная проверка: значения не должны быть слишком большими
                    if np.max(self.solution.y) < 10:  # Ограничиваем максимальное значение
                        print(f"Успешная генерация параметров с попытки {attempt + 1}")
                        return params

            except Exception as e:
                # Продолжаем попытки при ошибках
                continue

        # Если все попытки неудачны, создаем простые стабильные параметры
        print("Используем резервные параметры")
        return self.create_fallback_parameters()

    def create_fallback_parameters(self):
        """Создает простые стабильные параметры"""
        parameters = {}

        # Простые параметры K1-K15
        for i in range(1, 16):
            param_name = f"K{i}"
            min_r, max_r = self.pR[param_name]
            parameters[param_name] = min_r + 0.2 * (max_r - min_r)
            parameters[f"{param_name}_max"] = min_r + 0.8 * (max_r - min_r)

        # Простые значения R1-R4
        for i in range(1, 5):
            parameters[f"R{i}"] = 0.5

        # Очень простые полиномы
        for i in range(1, 56):
            parameters[f"f{i}_a3"] = 0.0
            parameters[f"f{i}_a2"] = 0.0
            parameters[f"f{i}_a1"] = 0.5  # Линейный рост
            parameters[f"f{i}_a0"] = 0.2  # Положительное начальное значение

        return parameters

    def _get_normalized(self, params):
        params_norm = params.copy()

        # Нормализация значений переменных K1-K15
        for i in range(1, 16):
            param_name = f"K{i}"
            min_val, max_val = self.pR[param_name]

            if param_name in params_norm:
                params_norm[param_name] = (params_norm[param_name] - min_val) / (max_val - min_val)

            if f"{param_name}_max" in params_norm:
                params_norm[f"{param_name}_max"] = (params_norm[f"{param_name}_max"] - min_val) / (max_val - min_val)

        return params_norm

    def polynomial_value(self, x, fn_index):
        a3 = self.parameters.get(f"f{fn_index}_a3", 0)
        a2 = self.parameters.get(f"f{fn_index}_a2", 0)
        a1 = self.parameters.get(f"f{fn_index}_a1", 0)
        a0 = self.parameters.get(f"f{fn_index}_a0", 0)

        # Ограничиваем входное значение для стабильности
        x_safe = max(0, min(1, x))
        return a3*x_safe**3 + a2*x_safe**2 + a1*x_safe + a0

    def disturbance_function(self, t, disturbance_index):
        """ОЧЕНЬ плавная функция возмущений - без волн и колебаний"""
        r_value = self.parameters.get(f"R{disturbance_index}", 0.5)

        # Масштабируем время от 0 до 1
        t_scaled = t / 10.0

        # Используем очень плавную сигмоидальную функцию
        # Более пологая сигмоида для плавного роста
        sigmoid = 1 / (1 + np.exp(-3 * (t_scaled - 0.5)))  # Еще более пологая

        # Плавный рост от начального значения
        start_value = 0.3 + r_value * 0.4  # Начинаем от 0.3 до 0.7
        growth = 0.2 * sigmoid  # Меньший рост для стабильности

        result = start_value + growth

        # Ограничиваем диапазон
        return max(0.3, min(0.9, result))

    def system_equations(self, t, K):
        try:
            dKdt = np.zeros(15)

            # Ограничиваем значения K для стабильности
            K_safe = np.maximum(0, np.minimum(1, K))

            # Внешние воздействия через функции возмущений
            R1 = self.disturbance_function(t, 1)
            R2 = self.disturbance_function(t, 2)
            R3 = self.disturbance_function(t, 3)
            R4 = self.disturbance_function(t, 4)

            # Уравнения системы с плавными изменениями и безопасными значениями
            dKdt[0] = - (self.polynomial_value(K_safe[9], 1) *
                         self.polynomial_value(K_safe[10], 2) *
                         self.polynomial_value(K_safe[13], 3)) * 0.05  # Уменьшил коэффициент

            dKdt[1] = (self.polynomial_value(K_safe[2], 4) *
                       self.polynomial_value(K_safe[6], 5) *
                       self.polynomial_value(K_safe[7], 6) *
                       self.polynomial_value(K_safe[8], 7) *
                       self.polynomial_value(K_safe[12], 8) -
                       (self.polynomial_value(K_safe[9], 9) *
                        self.polynomial_value(K_safe[10], 10) *
                        self.polynomial_value(K_safe[13], 11) *
                        self.polynomial_value(K_safe[14], 12) *
                        (R1 + R2 + R3 + R4))) * 0.03  # Уменьшил коэффициент

            dKdt[2] = (self.polynomial_value(K_safe[0], 13) -
                       (self.polynomial_value(K_safe[14], 14) * (R1 + R3 + R4))) * 0.05

            dKdt[3] = self.polynomial_value(K_safe[0], 15) * 0.05
            dKdt[4] = (self.polynomial_value(K_safe[0], 16) * R2 - R1) * 0.05
            dKdt[5] = (R2 - (self.polynomial_value(K_safe[3], 17) *
                             self.polynomial_value(K_safe[10], 18) *
                             self.polynomial_value(K_safe[11], 19) *
                             self.polynomial_value(K_safe[13], 20) * R1)) * 0.05

            dKdt[6] = (self.polynomial_value(K_safe[4], 21) *
                       self.polynomial_value(K_safe[5], 22) *
                       self.polynomial_value(K_safe[12], 23) *
                       self.polynomial_value(K_safe[14], 24) *
                       (R1 + R2 + R3)) * 0.03

            dKdt[7] = (self.polynomial_value(K_safe[4], 25) *
                       self.polynomial_value(K_safe[5], 26) *
                       self.polynomial_value(K_safe[10], 27) *
                       self.polynomial_value(K_safe[12], 28) *
                       self.polynomial_value(K_safe[13], 29) *
                       self.polynomial_value(K_safe[14], 30) *
                       (R1 + R2 + R3)) * 0.02

            dKdt[8] = (self.polynomial_value(K_safe[2], 31) *
                       self.polynomial_value(K_safe[12], 32) * R2 -
                       self.polynomial_value(K_safe[9], 33) *
                       self.polynomial_value(K_safe[10], 34) *
                       self.polynomial_value(K_safe[13], 35) * R1) * 0.05

            dKdt[9] = (self.polynomial_value(K_safe[2], 36) *
                       self.polynomial_value(K_safe[8], 37) *
                       self.polynomial_value(K_safe[14], 38) *
                       (R1 + R2 + R3 + R4)) * 0.03

            dKdt[10] = (self.polynomial_value(K_safe[2], 39) *
                        self.polynomial_value(K_safe[12], 40) *
                        self.polynomial_value(K_safe[13], 41) * (R1 + R3) -
                        self.polynomial_value(K_safe[14], 42) * R4) * 0.05

            dKdt[11] = (self.polynomial_value(K_safe[10], 43) *
                        self.polynomial_value(K_safe[12], 44) *
                        self.polynomial_value(K_safe[13], 45) * (R1 + R2 + R3) -
                        self.polynomial_value(K_safe[14], 46)) * 0.05

            dKdt[12] = (self.polynomial_value(K_safe[1], 47) *
                        self.polynomial_value(K_safe[2], 48) * R2) * 0.05

            dKdt[13] = (self.polynomial_value(K_safe[10], 49) *
                        self.polynomial_value(K_safe[11], 50) *
                        self.polynomial_value(K_safe[12], 51) * (R1 + R2)) * 0.05

            dKdt[14] = (self.polynomial_value(K_safe[1], 52) *
                        self.polynomial_value(K_safe[2], 53) *
                        self.polynomial_value(K_safe[12], 54) *
                        self.polynomial_value(K_safe[13], 55) * (R1 + R2)) * 0.03

            return dKdt

        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            return np.zeros(15)

    def solve_system(self):
        try:
            # начальные условия из параметров
            K0 = [self.parameters_norm[f"K{i}"] for i in range(1, 16)]

            # Увеличиваем количество точек для более плавных графиков
            self.time_points = np.linspace(0, 10, 1000)

            # решение системы с более строгими допусками
            self.solution = solve_ivp(
                self.system_equations,
                [0, 10],
                K0,
                t_eval=self.time_points,
                method='RK45',
                rtol=1e-8,  # Более строгий допуск
                atol=1e-10  # Более строгий допуск
            )

            # Нормализуем время для отображения от 0 до 1
            if hasattr(self, 'solution') and self.solution is not None:
                self.original_time = self.solution.t.copy()
                self.solution.t = self.solution.t / self.solution.t[-1]

            return self.solution

        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            # Возвращаем пустое решение вместо исключения
            self.solution = type('obj', (object,), {
                't': np.array([0, 1]),
                'y': np.ones((15, 2)) * 0.5
            })
            return self.solution

    def plot_time_series(self):
        if self.solution is None:
            self.solve_system()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, 15))
        groups = [(0, 5), (5, 10), (10, 15)]

        # ВСЕ линии делаем сплошными
        line_styles = ['-'] * 15

        # переменная для функций под графиками
        func_descriptions = []

        for ax_idx, (start, end) in enumerate(groups):
            ax = axes[ax_idx]
            for i in range(start, end):
                t = self.solution.t
                y = self.solution.y[i]

                # считаем функцию для кривой Kn по нормализованному времени
                coeffs = np.polyfit(t, y, deg=3)
                a3, a2, a1, a0 = coeffs
                func_descriptions.append(
                    f"K{i+1}(t) = {round(a3, 6)}·t³ + {round(a2, 6)}·t² + {round(a1, 6)}·t + {round(a0, 6)}"
                )

                y_label = self.names[f'K{i+1}']
                ax.plot(t, y, label=f'K{i+1} ({y_label})',
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
                        f'K{i+1}',
                        color='black',
                        fontsize=10,
                        fontweight='bold',
                        ha='left',
                        va='center',
                        alpha=0.9,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1)
                    )

            ax.set_title(f'K{start+1}–K{end}', fontsize=14)
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

        # выводим текст под графиками
        func_text = "\n".join(func_descriptions)
        fig.text(0.5, -0.08, func_text, ha='center', va='top', fontsize=12)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig)
        return img_b64

    def plot_disturbances(self):
        """График внешних возмущений - ОЧЕНЬ плавные кривые"""
        # Увеличиваем количество точек для максимальной плавности
        time_points = np.linspace(0, 1, 1000)

        fig, ax = plt.subplots(figsize=(12, 6))

        disturbance_names = [
            'R1 - Уровень финансирования системы МЧС города',
            'R2 - Степень экономического развития города',
            'R3 - Наличие прибыли у предприятия',
            'R4 - Доля современного оборудования на предприятии'
        ]
        disturbance_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i in range(1, 5):
            # Вычисляем значения для ОЧЕНЬ плавных кривых
            y = [self.disturbance_function(t * 10, i) for t in time_points]
            ax.plot(time_points, y, label=disturbance_names[i-1],
                    linewidth=3,  # Более толстые линии
                    color=disturbance_colors[i-1],
                    alpha=0.8)

        ax.legend(loc='best', fontsize=10)
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Значения возмущений', fontsize=12)
        ax.set_title('Динамика внешних возмущений', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.2, 1.0)  # Начинаем от 0.2 для лучшей визуализации

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
        categories = [f'K{i+1}' for i in range(15)]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axes = axes.flatten()

        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))

        for i, (t_idx, ax) in enumerate(zip(time_indices, axes)):
            if i >= len(time_points):
                break
            values = self.solution.y[:, t_idx].tolist()
            values += values[:1]

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid')
            ax.fill(angles, values, color=colors[i], alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_rlabel_position(0)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color='black', pad=10)

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