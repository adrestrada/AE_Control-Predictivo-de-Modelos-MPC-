import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def main():
    # Parámetros cinéticos de las reacciones
    k1 = 0.1
    k2 = 0.05
    k3 = 0.02

    # Ecuaciones cinéticas
    def dA_dt(t, variables):
        A, B, C, T, P = variables
        return -k1 * A * B

    def dB_dt(t, variables):
        A, B, C, T, P = variables
        return k1 * A * B - k2 * B * C

    def dC_dt(t, variables):
        A, B, C, T, P = variables
        return k2 * B * C - k3 * C

    # Temperatura, presión y catalizador (si es necesario)
    def dT_dt(t, variables):
        A, B, C, T, P = variables
        return -0.05 * T

    def dP_dt(t, variables):
        A, B, C, T, P = variables
        return 0.02

    # Sistema de ecuaciones diferenciales
    def equations(t, variables):
        dA = dA_dt(t, variables)
        dB = dB_dt(t, variables)
        dC = dC_dt(t, variables)
        dT = dT_dt(t, variables)
        dP = dP_dt(t, variables)
        return [dA, dB, dC, dT, dP]

    # Condiciones iniciales y valores iniciales
    initial_conditions = [1.0, 0.0, 0.0, 298.15, 1.0]
    t_span = (0, 10)

    # Resolver el sistema de ecuaciones diferenciales
    sol = solve_ivp(equations, t_span, initial_conditions, method='RK45')

    # Graficar los resultados
    plt.figure()
    plt.plot(sol.t, sol.y[0], label='A')
    plt.plot(sol.t, sol.y[1], label='B')
    plt.plot(sol.t, sol.y[2], label='C')
    plt.plot(sol.t, sol.y[3], label='T')
    plt.plot(sol.t, sol.y[4], label='P')

    plt.xlabel('Tiempo')
    plt.ylabel('Concentración / Temperatura / Presión')
    plt.legend()
    plt.grid()
    plt.title('Modelo de Reacciones Químicas con Factores Adicionales')
    plt.show()

if __name__ == "__main__":
    main()
