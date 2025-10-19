# pip install numpy scipy matplotlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random
import os
import io
import base64

class Calculator2:
    def __init__(self):
        self.names = {
            "X1": "Количество забракованных балок на 100 единиц продукции",
            "X2": "Численность операторов РТК",
            "X3": "Среднее количество остановок РТК на один цикл",
            "X4": "Средняя длина дефектных сварных швов на 1 единицу продукции",
            "X5": "Выполненные работы по плановому обслуживанию РТК",
            "X6": "Численность программистов",
            "X7": "Численность наладчиков сварочного оборудования",
            "X8": "Численность контролеров ОТК",
            "X9": "Численность цеховых технологов",
            "X10": "Количество дней просрочки поставки материалов и запчастей для ремонта РТК",
            "X11": "Среднее отклонение напряжения сварочной дуги",
            "X12": "Среднее отклонение тока на двигателе подающего блока",
            "X13": "Среднее отклонение манипулятора от программной траектории",
            "X14": "Наличие на рабочих местах необходимой технологической документации",
            "X15": "Отклонение давления защитного газа",
            "X16": "Отклонение давления сжатого воздуха",
            "X17": "План производства на заданный период в единицах продукции",
            "X18": "Количество балок, сданных ОТК с первого предъявления"
        }
        # временные точки
        self.time_points = np.linspace(0, 1, 100)
        # переменная для решения системы
        self.solution = None
        # диапазоны значений для параметров системы: (мин, макс, кол-во знаков после запятой)
        self.pR = {
            'X1': (0, 10, 0), 'X2': (0, 8, 0), 'X3': (0, 20, 0), 'X4': (0, 3, 6), 'X5': (0, 1, 0), 'X6': (0, 3, 0),
            'X7': (0, 4, 0), 'X8': (0, 2, 0), 'X9': (0, 1, 0), 'X10': (0, 30, 0), 'X11': (0, 5, 6), 'X12': (0, 2, 6),
            'X13': (0, 5, 6), 'X14': (0, 1, 0), 'X15': (0, 2, 6), 'X16': (0, 3, 6), 'X17': (0, 180, 0), 'X18': (0, 360, 0),
        }
        # константы системы
        self.constants = {
            'O0': (0, 8, 0), 'Oin': (0, 2, 0), 'Oout': (0, 2, 0),
            'Sm': (1, 3, 0), 'Rw': (1, 6, 0), 'Nst': (0, 500, 0),
            'S_star': (0, 5, 0), 'Ld': (0, 120, 6), 'L_star': (0, 80, 6),
            'Mf': (0, 22, 0), 'Mp': (0, 22, 0), 'P0': (0, 3, 0),
            'Pin': (0, 1, 0), 'Pout': (0, 1, 0), 'R0': (0, 4, 0),
            'Rin': (0, 1, 0), 'Rout': (0, 1, 0), 'C0': (0, 2, 0),
            'Cin': (0, 1, 0), 'Cout': (0, 1, 0), 'T0': (0, 1, 0),
            'Tin': (0, 1, 0), 'Tout': (0, 1, 0), 'Nr': (0, 3, 6),
            'Df': (0, 30, 0), 'Dp': (0, 30, 0), 'DeltaU': (0, 5, 6),
            'Delta_star_U': (0, 2, 6), 'DeltaI': (0, 2, 6),
            'Delta_star_I': (0, 0.8, 6), 'DeltaT': (0, 5, 6),
            'Delta_star_T': (0, 1, 6), 'Tdf': (0, 12, 0),
            'Tdp': (0, 12, 0), 'DeltaPG': (0, 2, 6),
            'Delta_star_PG': (0, 0.5, 6), 'DeltaPV': (0, 3, 6),
            'Delta_star_PV': (0, 1, 6), 'NTP': (0, 360, 0),
            'Nd': (0, 360, 0), 'Ab': (0, 50, 0),
        }
        # все параметры системы (словарь)
        self.parameters = self.generate_valid_parameters()
        self.parameters_norm = {}

    # генерация словаря случайных значений параметров системы   
    def generate_system_parameters(self):        
        parameters = {}
        
        # параметры X1-X18
        for i in range(1, 19):
            param_name = f"X{i}"
            min_r, max_r, dec = self.pR[param_name]
            base_min_val = min_r + ((max_r - min_r) * 0.05)
            base_value = round(random.uniform(base_min_val, max_r), dec)
            min_value = round(random.uniform(min_r, base_value), dec) if base_value > min_r else min_r
            max_value = round(random.uniform(base_value, max_r), dec) if base_value < max_r else max_r
            parameters[param_name] = base_value
            parameters[f"{param_name}_min"] = min_value
            parameters[f"{param_name}_max"] = max_value
        
        # константы
        for const_name, (min_val, max_val, dec_places) in self.constants.items():
            parameters[const_name] = round(random.uniform(min_val, max_val), dec_places)
        
        # коэффициенты для полиномиальных функций f1-f36
        for i in range(1, 37):           
            parameters[f"f{i}_a3"] = round(random.uniform(0.0, 0.3), 6)
            parameters[f"f{i}_a2"] = round(random.uniform(0.0, 0.4), 6)
            parameters[f"f{i}_a1"] = round(random.uniform(0.1, 0.6), 6)
            parameters[f"f{i}_a0"] = round(random.uniform(0.0, 0.2), 6)
        
        return parameters

    def generate_valid_parameters(self):
        # максимум 10 попыток
        for _ in range(10):  
            params = self.generate_system_parameters()
            params_norm = self._get_normalized(params)
            self.parameters = params
            self.parameters_norm = params_norm
            try:                
                self.solve_system()
                if np.all(self.solution.y >= -1e-6):                    
                    return params
            except:
                continue
        raise ValueError("Не удалось сгенерировать физически корректные параметры после 10 попыток")
    
    def _get_normalized(self, params):
        params_norm = params.copy()        
        # Нормализация значений переменных от 0 до 1
        for i in range(18):
            param_name = f"X{i+1}"
            min_val, max_val, _ = self.pR[param_name]
            params_norm[param_name] = (params_norm[param_name] - min_val) / (max_val - min_val)
            params_norm[f"{param_name}_min"] = (params_norm[f"{param_name}_min"] - min_val) / (max_val - min_val)
            params_norm[f"{param_name}_max"] = (params_norm[f"{param_name}_max"] - min_val) / (max_val - min_val)

        for const_name, (min_val, max_val, _) in self.constants.items():
            params_norm[const_name] = (params_norm[const_name] - min_val) / (max_val - min_val)   
        
        return params_norm
    
    # вычисление значения полинома f_n(x) = a3*x^3 + a2*x^2 + a1*x + a0
    def polynomial_value(self, x, fn_index):   
        a3 = self.parameters.get(f"f{fn_index}_a3", 0)
        a2 = self.parameters.get(f"f{fn_index}_a2", 0)
        a1 = self.parameters.get(f"f{fn_index}_a1", 0)
        a0 = self.parameters.get(f"f{fn_index}_a0", 0)
        
        return a3*x**3 + a2*x**2 + a1*x + a0
    
    # система дифференциальных уравнений
    def system_equations(self, t, X):
        try:
            dXdt = np.zeros(18)
            params = self.parameters_norm

            # уравнения системы
            dXdt[0] = params['Rw'] * self.polynomial_value(X[2], 1) * self.polynomial_value(X[10], 2) * \
                      self.polynomial_value(X[11], 3) * self.polynomial_value(X[12], 4) - \
                      params['Nst'] * self.polynomial_value(X[1], 5) * self.polynomial_value(X[7], 6) * \
                      self.polynomial_value(X[16], 7)
            
            dXdt[1] = (params['O0'] + params['Oin']) * self.polynomial_value(X[16], 12) - \
                      (params['Sm'] + params['Rw'] + params['Oout'])
            
            dXdt[2] = params['Nst'] / max(params['Rw'], 0.001) * self.polynomial_value(X[9], 8) * \
                      self.polynomial_value(X[14], 9) * self.polynomial_value(X[15], 10) - \
                      params['S_star'] * self.polynomial_value(X[1], 11)
            
            dXdt[3] = params['Ld'] * self.polynomial_value(X[14], 13) * self.polynomial_value(X[15], 14) - \
                      params['L_star'] * self.polynomial_value(X[1], 15)
            
            dXdt[4] = params['Mf'] * self.polynomial_value(X[5], 16) * self.polynomial_value(X[6], 17) - \
                      params['Mp'] * self.polynomial_value(X[9], 18)
            
            dXdt[5] = (params['P0'] + params['Pin']) * self.polynomial_value(X[16], 19) - \
                      (params['Sm'] + params['Rw'] + params['Pout'])
            
            dXdt[6] = (params['R0'] + params['Rin']) * self.polynomial_value(X[16], 20) - \
                      (params['Sm'] + params['Rw'] + params['Rout'])
            
            dXdt[7] = (params['C0'] + params['Cin']) * self.polynomial_value(X[16], 21) - \
                      (params['Sm'] + params['Rw'] + params['Cout'])
            
            dXdt[8] = (params['T0'] + params['Tin']) * self.polynomial_value(X[16], 22) - params['Tout']
            
            dXdt[9] = (params['Nr'] + params['Df']) * self.polynomial_value(X[16], 23) - params['Dp']
            
            dXdt[10] = params['DeltaU'] - params['Delta_star_U'] * self.polynomial_value(X[4], 24)
            
            dXdt[11] = params['DeltaI'] - params['Delta_star_I'] * self.polynomial_value(X[4], 25)
            
            dXdt[12] = params['DeltaT'] - params['Delta_star_T'] * self.polynomial_value(X[4], 26)
            
            dXdt[13] = params['Tdf'] * self.polynomial_value(X[8], 27) - params['Tdp']
            
            dXdt[14] = params['DeltaPG'] - params['Delta_star_PG'] * self.polynomial_value(X[16], 28)
            
            dXdt[15] = params['DeltaPV'] - params['Delta_star_PV'] * self.polynomial_value(X[16], 29)
            
            dXdt[16] = params['NTP'] * self.polynomial_value(X[8], 30) - params['Rw']
            
            dXdt[17] = params['Nd'] * self.polynomial_value(X[5], 31) * self.polynomial_value(X[6], 32) * \
                       self.polynomial_value(X[7], 33) * self.polynomial_value(X[13], 34) - \
                       (params['Ab'] + params['Ld']) * self.polynomial_value(X[0], 35) * self.polynomial_value(X[3], 36)
            
            dXdt = np.clip(dXdt, 0, None)  
            return dXdt
            
        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            return np.zeros(18)
    
    # решение системы дифференциальных уравнений
    def solve_system(self):        
        try:
            # начальные условия из параметров
            X0 = [self.parameters_norm[f"X{i}"] for i in range(1, 19)]
            
            # решение системы
            self.solution = solve_ivp(
                self.system_equations, 
                [0, 1], 
                X0, 
                t_eval=self.time_points, 
                method='RK45',
                rtol=1e-6,
                atol=1e-8
            )
            self.solution.y = np.clip(self.solution.y, 0, None)                        
            return self.solution
            
        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            return []

    def plot_time_series(self):           
        if self.solution is None:
            self.solve_system()
        # создаем график, состоящий из трех подграфиков
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        colors = plt.cm.tab20(np.linspace(0, 1, 18))
        groups = [(0, 6), (6, 12), (12, 18)]

        # переменная для функций под графиками
        func_descriptions = []

        for ax_idx, (start, end) in enumerate(groups):
            ax = axes[ax_idx]
            for i in range(start, end):
                t = self.solution.t
                y = self.solution.y[i]
                # считаем функцию для кривой Хn  
                coeffs = np.polyfit(t, y, deg=3)
                a3, a2, a1, a0 = coeffs              
                func_descriptions.append(
                    f"X{i+1}(t) = {round(a3, 6)}·t³ + {round(a2, 6)}·t² + {round(a1, 6)}·t + {round(a0, 6)}"
                )   
                # рисуем кривую, обрезаем ее значение, если оно стало больше 1.15 по у, и подписываем ее
                y = [val for val in self.solution.y[i] if val <= 1.15]
                y_label = self.names[f'X{i+1}']
                ax.plot(self.solution.t[:len(y)], y, label=f'X{i+1} ({y_label})', color=colors[i], linewidth=2)
                if len(y) > 0:
                    ax.text(self.solution.t[len(y)-1]+0.015, y[-1]+0.015, f'X{i+1}', color="black",
                        fontsize=10, va='center')
                    
            ax.set_title(f'X{start+1}–X{end}', fontsize=14)
            ax.set_xlabel('Время', fontsize=10)
            if ax_idx == 0:
                ax.set_ylabel('Значение параметра', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.2)

        # объединяем все легенды
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l

        # общая легенда со всеми X1–X18, в несколько столбцов
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
    
    # отрисовка 5 лепестковых диаграмм
    def plot_radar_charts(self):
        if self.solution is None:
            self.solve_system()

        time_points = [0, 0.25, 0.5, 0.75, 1]
        time_indices = [np.abs(self.solution.t - t).argmin() for t in time_points]
        categories = [f'X{i+1}' for i in range(18)]
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw=dict(polar=True))
        axes = axes.flatten()
        colors = plt.cm.viridis(np.linspace(0, 1, len(time_points)))
        # красный контур — максимальные пределы
        max_values = [self.parameters_norm.get(f"X{i+1}_max", 1) for i in range(18)]
        max_values += max_values[:1]
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
            ax.plot(angles, max_values, color='red', linewidth=2, linestyle='dashed', label='Максимум')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_rlabel_position(0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color=colors[i], pad=10)
        for i in range(len(time_points), len(axes)):
            fig.delaxes(axes[i])
        plt.tight_layout()
        radar_buffer = io.BytesIO()
        plt.savefig(radar_buffer, format='png', dpi=100, bbox_inches='tight')
        radar_buffer.seek(0)
        radar_b64 = base64.b64encode(radar_buffer.getvalue()).decode()
        plt.close(fig)
        return radar_b64
    
    # построение всех графиков
    def plot_all_results(self):
        self.solve_system()
        time_series_b64 = self.plot_time_series()
        radar_b64 = self.plot_radar_charts()
        return [time_series_b64, radar_b64]