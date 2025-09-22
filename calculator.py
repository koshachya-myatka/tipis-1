# pip install numpy scipy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

class Calculator:
    def __init__(self):
        # все параметры системы (словарь)
        self.parameters = self.generate_system_parameters()
        # временные точки от 0 до 1 с шагом 0.01
        self.time_points = np.linspace(0, 1, 100)
        # переменная для решения системы
        self.solution = None

    # генерация словаря случайных значений параметров системы   
    def generate_system_parameters(self):
        parameters = {}
        
        # параметры X1-X18 (от 0.01 до 1.00)
        for i in range(1, 19):
            param_name = f"X{i}"
            parameters[param_name] = round(random.uniform(0.01, 1.0), 2)
            parameters[f"{param_name}_min"] = round(random.uniform(0.01, parameters[param_name]), 2)
        
        # константы системы
        constants = {
            'O0': (1, 5), 'Oin': (1, 3), 'Oout': (1, 3),
            'Sm': (1, 2), 'Rw': (1, 5), 'Nst': (1, 10),
            'S_star': (1, 3), 'Ld': (1, 10), 'L_star': (1, 5),
            'Mf': (1, 10), 'Mp': (1, 10), 'P0': (1, 5),
            'Pin': (1, 2), 'Pout': (1, 2), 'R0': (1, 5),
            'Rin': (1, 2), 'Rout': (1, 2), 'C0': (1, 5),
            'Cin': (1, 2), 'Cout': (1, 2), 'T0': (1, 5),
            'Tin': (1, 2), 'Tout': (1, 2), 'Nr': (1, 5),
            'Df': (1, 10), 'Dp': (1, 5), 'DeltaU': (1, 5),
            'Delta_star_U': (1, 3), 'DeltaI': (1, 5),
            'Delta_star_I': (1, 3), 'DeltaT': (1, 5),
            'Delta_star_T': (1, 3), 'Tdf': (1, 10),
            'Tdp': (1, 10), 'DeltaPG': (1, 5),
            'Delta_star_PG': (1, 3), 'DeltaPV': (1, 5),
            'Delta_star_PV': (1, 3), 'NTP': (1, 20),
            'Nd': (1, 20), 'Ab': (1, 10),
        }
        
        for const_name, (min_val, max_val) in constants.items():
            parameters[const_name] = random.randint(min_val, max_val)
        
        # коэффициенты для полиномиальных функций f1-f36
        for i in range(1, 37):
            parameters[f"f{i}_a3"] = round(random.uniform(0.1, 10.0), 2)
            parameters[f"f{i}_a2"] = round(random.uniform(0.1, 10.0), 2)
            parameters[f"f{i}_a1"] = round(random.uniform(0.1, 10.0), 2)
            parameters[f"f{i}_a0"] = round(random.uniform(0.1, 10.0), 2)
        
        return parameters
    
    # вычисление значения полинома f_n(x) = a3*x^3 + a2*x^2 + a1*x + a0
    # в параметрах (значение икса, n-номер для fn)
    def polynomial_value(self, x, fn_index):        
        a3 = self.parameters.get(f"f{fn_index}_a3", 0)
        a2 = self.parameters.get(f"f{fn_index}_a2", 0)
        a1 = self.parameters.get(f"f{fn_index}_a1", 0)
        a0 = self.parameters.get(f"f{fn_index}_a0", 0)
        
        # ограничиваем значение x для избежания переполнения
        x = np.clip(x, 0, 1)  # X всегда от 0 до 1
        
        return a3*x**3 + a2*x**2 + a1*x + a0
    
    # система дифференциальных уравнений
    def system_equations(self, t, X):
        try:
            dXdt = np.zeros(18)
            params = self.parameters
            
            # ограничиваем значения X от 0 до 1
            X = np.clip(X, 0, 1)

            # сами функции производных для Х1-Х18, 
            # параметр Хn указывается с индексом на 1 меньше, чем на картинке
            # номер функции f соответствует картинке
            
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
            
            # ограничиваем производные для стабильности и положительных значений
            dXdt = np.clip(dXdt, -0.1, 0.1)
            
            return dXdt
            
        except Exception as e:
            print(f"Ошибка в вычислении производных: {e}")
            return np.zeros(18)
    
    # решение системы дифференциальных уравнений
    def solve_system(self):
        try:
            # начальные условия из параметров, список значени Х1-Х18 на момент времени 0
            X0 = [self.parameters[f"X{i}"] for i in range(1, 19)]
            
            # решение системы
            self.solution = solve_ivp(
                self.system_equations, 
                [0, 1], 
                X0, 
                t_eval=self.time_points, 
                method='RK45',
                rtol=1e-3,
                atol=1e-3
            )
            
            # срез значений по оси y от 0 до 1
            self.solution.y = np.clip(self.solution.y, 0, 1)
            
            return self.solution
            
        except Exception as e:
            print(f"Ошибка при решении системы: {e}")
            # Возвращаем фиктивное решение для отладки
            self.solution = type('obj', (object,), {
                't': self.time_points,
                'y': np.random.uniform(0, 1, (18, len(self.time_points)))
            })
            return self.solution
        
    # отрисовка графика изменений Х1-Х18 по времени
    def plot_time_series(self):
        if self.solution is None:
            self.solve_system()
        
        plt.figure(figsize=(12, 8))
        
        for i in range(18):
            plt.plot(self.solution.t, self.solution.y[i], label=f'X{i+1}', linewidth=2)
        
        plt.xlabel('Время', fontsize=12)
        plt.ylabel('Значение параметра', fontsize=12)
        plt.title('Изменение параметров X1-X18 по времени', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)  # Ограничиваем ось Y от 0 до 1
        
        plt.tight_layout()
        plt.show()
    
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
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'Время t = {self.solution.t[t_idx]:.2f}', size=11, color=colors[i], pad=10)
        for i in range(len(time_points), len(axes)):
            fig.delaxes(axes[i])
        plt.suptitle('Лепестковые диаграммы параметров X1-X18', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    # построение всех графиков
    def plot_all_results(self):
        self.plot_time_series()
        self.plot_radar_charts()