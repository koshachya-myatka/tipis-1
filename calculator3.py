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
            # Генерируем случайные значения в допустимых диапазонах
            base_value = round(random.uniform(min_r, max_r * 0.3), 2)  # Более консервативные значения
            max_value = round(random.uniform(base_value, max_r), 2)    # Макс >= базовому
            parameters[param_name] = base_value
            parameters[f"{param_name}_max"] = max_value

        # Внешние воздействия R1-R4
        for i in range(1, 5):
            parameters[f"R{i}"] = round(random.uniform(0.3, 0.8), 3)  # Более узкий диапазон для стабильности

        # Полиномы для внутренних функций f1-f55 - более консервативные коэффициенты
        for i in range(1, 56):
            # Делаем полиномы более простыми и стабильными
            parameters[f"f{i}_a3"] = round(random.uniform(-0.5, 0.5), 3)   # Меньший разброс
            parameters[f"f{i}_a2"] = round(random.uniform(-0.8, 0.8), 3)
            parameters[f"f{i}_a1"] = round(random.uniform(0.2, 0.8), 3)    # Положительный наклон
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 0.4), 3)    # Положительное смещение

        return parameters

    def generate_valid_parameters(self):
        # Пробуем сгенерировать параметры несколько раз
        for attempt in range(10):
            try:
                params = self.generate_system_parameters()
                params_norm = self._get_normalized(params)
                self.parameters = params
                self.parameters_norm = params_norm

                # Пробуем решить систему
                solution = self.solve_system()

                # Более мягкие проверки стабильности
                if (solution is not None and
                        np.all(np.isfinite(solution.y)) and
                        not np.any(np.isnan(solution.y)) and
                        np.max(solution.y) < 5.0):  # Менее строгое ограничение

                    print(f"Успешная генерация параметров с попытки {attempt + 1}")
                    return params

            except Exception as e:
                print(f"Попытка {attempt + 1} не удалась: {e}")
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
            # Используем более консервативные значения
            parameters[param_name] = round(min_r + 0.1 * (max_r - min_r), 2)
            parameters[f"{param_name}_max"] = round(min_r + 0.6 * (max_r - min_r), 2)

        # Простые значения R1-R4 - разные значения для разнообразия
        parameters["R1"] = 0.6
        parameters["R2"] = 0.5
        parameters["R3"] = 0.7
        parameters["R4"] = 0.4

        # Более разнообразные полиномы
        for i in range(1, 56):
            # Делаем полиномы немного разными
            parameters[f"f{i}_a3"] = 0.0
            parameters[f"f{i}_a2"] = round(random.uniform(-0.2, 0.2), 3)
            parameters[f"f{i}_a1"] = round(random.uniform(0.3, 0.7), 3)
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 0.3), 3)

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

    def polynomial_value(self, x, i):
        x = np.clip(x, -10, 10)

        a3 = self.parameters.get(f"f{i}_a3", 0)
        a2 = self.parameters.get(f"f{i}_a2", 0)
        a1 = self.parameters.get(f"f{i}_a1", 0.5)
        a0 = self.parameters.get(f"f{i}_a0", 0.2)

        return a3 * x**3 + a2 * x**2 + a1 * x + a0

    def disturbance_function(self, t, i):
        r = self.parameters.get(f"R{i}", 0.5)

        # Нормируем время
        x = np.clip(t / 10.0, 0, 1)

        # Плавный нелинейный рост
        s = 1 / (1 + np.exp(-6 * (x - 0.5)))

        # Стабильная вариативность
        curve = s * (0.8 + 0.2 * r)

        # Легкая асимметрия
        curve += 0.03 * (i - 2.5) * x * (1 - x)

        return float(np.clip(curve, 0, 1))

    def system_equations(self, t, K):
        try:
            dKdt = np.zeros(15)

            # Теперь НЕ clip
            K_safe = K

            # Внешние воздействия
            R1 = self.disturbance_function(t, 1)
            R2 = self.disturbance_function(t, 2)
            R3 = self.disturbance_function(t, 3)
            R4 = self.disturbance_function(t, 4)

            R_sum = (R1 + R2 + R3 + R4) / 4
            R_mix = 0.4 * R_sum + 0.2 * np.sin(t / 3)

            P = lambda x, idx: self.polynomial_value(x, idx)

            # Упрощенная и более стабильная система уравнений
            # K1
            dKdt[0] = 0.15 * P(K_safe[1], 1) - 0.1 * P(K_safe[2], 2) + 0.12 * R_mix

            # K2
            dKdt[1] = 0.18 * P(K_safe[0], 3) - 0.11 * P(K_safe[3], 4) + 0.1 * R1

            # K3
            dKdt[2] = 0.16 * P(K_safe[4], 5) - 0.14 * P(K_safe[1], 6) * R2 + 0.08 * R_mix

            # K4
            dKdt[3] = 0.14 * P(K_safe[0], 7) + 0.1 * P(K_safe[2], 8) - 0.09 * R3

            # K5
            dKdt[4] = 0.16 * P(K_safe[3], 9) - 0.12 * P(K_safe[7], 10) + 0.12 * R2

            # K6
            dKdt[5] = 0.15 * P(K_safe[1], 11) - 0.16 * P(K_safe[8], 12) + 0.11 * R_mix

            # K7
            dKdt[6] = 0.18 * P(K_safe[2], 13) - 0.14 * P(K_safe[5], 14) + 0.1 * R1

            # K8
            dKdt[7] = 0.17 * P(K_safe[6], 15) - 0.15 * P(K_safe[4], 16) + 0.12 * R3

            # K9
            dKdt[8] = 0.16 * P(K_safe[0], 17) - 0.13 * P(K_safe[10], 18) + 0.1 * R2

            # K10
            dKdt[9] = 0.15 * P(K_safe[8], 19) + 0.14 * P(K_safe[11], 20) - 0.12 * R4

            # K11
            dKdt[10] = 0.18 * P(K_safe[9], 21) - 0.16 * P(K_safe[12], 22) + 0.13 * R_mix

            # K12
            dKdt[11] = 0.17 * P(K_safe[10], 23) - 0.15 * P(K_safe[6], 24) + 0.11 * R1

            # K13
            dKdt[12] = 0.18 * P(K_safe[13], 25) - 0.16 * P(K_safe[5], 26) + 0.12 * R3

            # K14
            dKdt[13] = 0.17 * P(K_safe[11], 27) + 0.15 * P(K_safe[14], 28) - 0.1 * R2

            # K15
            dKdt[14] = 0.19 * P(K_safe[7], 29) - 0.16 * P(K_safe[9], 30) + 0.13 * R4

            # ✔ плавное приближение к границам
            for j in range(15):
                B = 4 * K[j] * (1 - K[j])          # барьер
                dKdt[j] *= B                       # подавление скорости у границ

            return dKdt

        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            return np.zeros(15)

    def solve_system(self):
        try:
            # начальные условия из параметров
            K0 = [self.parameters_norm.get(f"K{i}", 0.3) for i in range(1, 16)]

            self.time_points = np.linspace(0, 10, 500)  # Меньше точек для скорости

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

    # Остальные методы (plot_time_series, plot_disturbances, plot_radar_charts) остаются без изменений
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

                # Аппроксимация полиномом
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
        """График внешних возмущений"""
        time_points = np.linspace(0, 1, 500)

        fig, ax = plt.subplots(figsize=(12, 6))

        disturbance_names = [
            'R1 - Уровень финансирования системы МЧС города',
            'R2 - Степень экономического развития города',
            'R3 - Наличие прибыли у предприятия',
            'R4 - Доля современного оборудования на предприятии'
        ]
        disturbance_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        for i in range(1, 5):
            y = [self.disturbance_function(t * 10, i) for t in time_points]
            ax.plot(time_points, y, label=disturbance_names[i-1],
                    linewidth=3,
                    color=disturbance_colors[i-1],
                    alpha=0.8)

        ax.legend(loc='best', fontsize=10)
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Значения возмущений', fontsize=12)
        ax.set_title('Динамика внешних возмущений', fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.2, 1.0)

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