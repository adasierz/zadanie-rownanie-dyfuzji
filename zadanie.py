import numpy as np
import matplotlib.pyplot as plt

indeks = str(180954)
T1 = int(indeks[-2:])
T2 = int(indeks[0:3])

L = 1
D = 0.00001

h = 0.1
l = (h**2)/(4*D)

x_max = L
t_max = 60000

def obliczenia(p):
    if p == 0 or p == 1:
        alfa = (D * l) / (h ** 2)
        x = np.arange(0, x_max + h, h)
        t = np.arange(0, t_max + l, l)
        r = len(t)
        c = len(x)
        u = np.zeros([r, c])
        delta = 0
        deltapom = []
        for j in range(0, r-1):
            for i in range(0, c-1):
                u[:, 0] = T1
                u[:, 10] = T2
                u[j+1, i] = alfa*(u[j][i+1]+u[j][i-1])+(1-2*alfa)*u[j][i]
        delta += abs((u[j][i] - u[j - 1][i]) / u[j][i])
        delta /= 11
        for j in range(1, r):
            deltapom.append(abs((u[j][i] - u[j - 1][i]) / u[j][i]))
        if p == 0:
            return u
        if p == 1:
            print("Średni błąd względny wynosi:", delta)
            return deltapom

def wykres(p):
    if p == 0:
        x = np.arange(0, x_max + h, h)
        y = np.arange(0, t_max + l, l)
        z = obliczenia(0)
        X, Y = np.meshgrid(x, y, sparse=True)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_xlabel("x[m]")
        ax.set_ylabel("t[s]")
        ax.set_zlabel("T[st. C]")
        ax.set_title(f"Zależność temperatury od współrzędnych powierzchniowych i czasowych")
        plt.show()
    if p == 1:
        x = np.arange(250, t_max+l, l).tolist()
        y = obliczenia(1)
        plt.xlabel("czas t[s]", fontsize=11)
        plt.ylabel("błąd względny, delta[st. C]", fontsize=11)
        plt.xlim(0, 60000, l)
        plt.ylim(0, 1)
        plt.plot(x, y, color="blue")
        plt.show()

def tabela():
    np.savetxt("dane", obliczenia(0), fmt="%10f", delimiter='\t')

wykres(0)
tabela()

wykres(1)