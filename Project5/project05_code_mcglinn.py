import cmath
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

def quantum_it(d, b=0):

    # The space grid along the x-axis has K_space points
    if b == 2:
        K_space = 3000
    else:
        K_space = 2500

    # The time grid has N_time steps
    N_time = 2048

    # Space-time grid spacings
    dx = 5 * 10 ** -4
    dt = 5 * 10 ** -7

    # Wave packet parameters
    if b == 2:
        sigma = np.sqrt(0.004)
    else:
        sigma = np.sqrt(2.5e-3)
    k0 = 620
    x0 = 0.3

    # Barrier parameters
    bar_thick = d  # thickness
    v0 = 9.8e5  # height

    # VARIABLES:

    # TDSE variable
    q = 0

    # Arrays
    x = np.zeros(K_space)
    V = np.zeros(K_space)

    # Wave function Psi(x,t) evaluated on discrete space-time grid is stored as complex array
    Psi = np.zeros((K_space, N_time), dtype=complex)

    # Auxilarry complex matrices of discrete Time-Dependent Schrodinger Equation
    A = np.zeros((K_space, K_space), dtype=complex)
    B = np.zeros((K_space, K_space), dtype=complex)

    # TDSE parameter:
    q = dt / (4 * dx * dx)

    # Evaluate coordinates of the discrete space grid:
    x[0] = 0
    for k in range(0, K_space - 1):
        x[k + 1] = x[k] + dx

    # Evaluate potential V(x) on the space grid x(k) and load into array V(1:K_space)
    for k in range(0, K_space):
        if x[k] > 0.6 and (x[k] < 0.6 + bar_thick) and b > 0:
            V[k] = v0
        elif x[k] > 0.6 + 0.004 and (x[k] < 0.6 + bar_thick + 0.004) and b == 2:
            V[k] = v0

    Psi[:, 0] = (1 / (((2.0 * np.pi) ** 0.25) * (sigma ** 0.5))) * np.exp(-(x - x0) ** 2 / (4 * sigma ** 2)) * np.exp(1j * k0 * (x - x0))

    #Fill in the following information
    A[0,0] = 1 + 2j*q
    B[0,0] = 1 - 2j*q

    for k in range(1,K_space):
        A[k,k] = 1 + 2j * q + 1j * dt / 2 * V[k]
        A[k-1, k] = -1j * q
        A[k, k-1] = -1j * q
        B[k,k] = 1 - 2j * q - 1j * dt / 2 * V[k]
        B[k-1, k] = 1j * q
        B[k, k-1] = 1j * q

    # Solve TDSE A*Psi(n+1)=B*Psi(n) for N_time-1 steps starting from the minimum
    # uncertainty (Gaussian wave packet) state loaded above at t=0, time step n=1:

    Asp = sp.csr_matrix(A, dtype=np.cfloat)
    Bsp = sp.csr_matrix(B, dtype=np.cfloat)

    for n in range(0, N_time - 1):
        Psi[:, n + 1] = np.linalg.inv(A).dot(Bsp.dot(Psi[:, n]))

    # Select wave function a time step n and plot the probability density defined by it
    np.savez("arrays2", Psi, V, x, A, B, Asp, Bsp)


def plot_it():
    File = np.load("arrays.npz")
    File2 = np.load("arrays2.npz")
    Fileb01 = np.load("arrays_testbarrier.npz")
    Fileb05 = np.load("arrays_barrier0.005.npz")
    Fileb1 = np.load("arrays_barrier0.01.npz")
    File2b = np.load("arrays_2barrier.npz")
    File2b2 = np.load("arrays_2barrier2.npz")

    Psi = File['arr_0']
    x = File['arr_8']
    Psi2 = File2['arr_0']
    x2 = File2['arr_2']
    Psib01 = Fileb01['arr_0']
    xb01 = Fileb01['arr_2']
    Vb01 = Fileb01['arr_1']
    Psib05 = Fileb05['arr_0']
    xb05 = Fileb05['arr_2']
    Vb05 = Fileb05['arr_1']
    Psib1 = Fileb1['arr_0']
    xb1 = Fileb1['arr_2']
    Vb1 = Fileb1['arr_1']
    Psi2b = File2b['arr_0']
    x2b = File2b['arr_2']
    V2b = File2b['arr_1']
    Psi2b2 = File2b2['arr_0']
    x2b2 = File2b2['arr_2']
    V2b2 = File2b2['arr_1']

    size = Psi.shape[1]

    n1 = 1
    n2 = size // 2
    n3 = size - 1

    rho1 = np.conj(Psi[:, n1])*Psi[:, n1]
    rho2 = np.conj(Psi[:, n2])*Psi[:, n2]
    rho3 = np.conj(Psi[:, n3])*Psi[:, n3]

    rho12 = np.conj(Psi2[:, n1]) * Psi2[:, n1]
    rho22 = np.conj(Psi2[:, n2]) * Psi2[:, n2]
    rho32 = np.conj(Psi2[:, n3]) * Psi2[:, n3]

    rhob01 = np.conj(Psib01[:, n1]) * Psib01[:, n1]
    rhob012 = np.conj(Psib01[:, n2]) * Psib01[:, n2]
    rhob013 = np.conj(Psib01[:, n3]) * Psib01[:, n3]

    rhob05 = np.conj(Psib05[:, n1]) * Psib05[:, n1]
    rhob052 = np.conj(Psib05[:, n2]) * Psib05[:, n2]
    rhob053 = np.conj(Psib05[:, n3]) * Psib05[:, n3]

    rhob1 = np.conj(Psib1[:, n1]) * Psib1[:, n1]
    rhob12 = np.conj(Psib1[:, n2]) * Psib1[:, n2]
    rhob13 = np.conj(Psib1[:, n3]) * Psib1[:, n3]

    rho2b = np.conj(Psi2b[:, n1]) * Psi2b[:, n1]
    rho2b2 = np.conj(Psi2b[:, n2]) * Psi2b[:, n2]
    rho2b3 = np.conj(Psi2b[:, n3]) * Psi2b[:, n3]

    rho2b21 = np.conj(Psi2b2[:, n1]) * Psi2b2[:, n1]
    rho2b22 = np.conj(Psi2b2[:, n2]) * Psi2b2[:, n2]
    rho2b32 = np.conj(Psi2b2[:, n3]) * Psi2b2[:, n3]

    fig, ax = plt.subplots(2,1)
    ax[0].plot(x2, Vb01)
    ax[1].plot(x2b, V2b)
    ax[0].set_xlim(0.55, 0.65)
    ax[1].set_xlim(0.55, 0.65)
    fig.supylabel("V")
    fig.supxlabel("x")
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x, rho1, label=r'$P$')
    ax[0].plot(x, np.real(Psi[:, n1]), label='Real')
    ax[0].plot(x, np.imag(Psi[:, n1]), label='Imaginary')
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].plot(x, rho2)
    ax[1].plot(x, np.real(Psi[:, n2]))
    ax[1].plot(x, np.imag(Psi[:, n2]))
    ax[2].plot(x, rho3)
    ax[2].plot(x, np.real(Psi[:, n3]))
    ax[2].plot(x, np.imag(Psi[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    fig.legend(handles, labels, loc='upper right')
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x2, rho12, label=r'$P$')
    ax[0].plot(x2, np.real(Psi2[:, n1]), label='Real')
    ax[0].plot(x2, np.imag(Psi2[:, n1]), label='Imaginary')
    handles, labels = ax[0].get_legend_handles_labels()
    ax[1].plot(x2, rho22)
    ax[1].plot(x2, np.real(Psi2[:, n2]))
    ax[1].plot(x2, np.imag(Psi2[:, n2]))
    ax[2].plot(x2, rho32)
    ax[2].plot(x2, np.real(Psi2[:, n3]))
    ax[2].plot(x2, np.imag(Psi2[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    fig.legend(handles, labels, loc='upper right')
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x2, rhob01)
    #ax[0].plot(x2, Vb01)
    #ax[0].plot(x2, np.real(Psi2[:, n1]))
    #ax[0].plot(x2, np.imag(Psi2[:, n1]))
    ax[1].plot(x2, rhob012)
    #ax[1].plot(x2, Vb01)
    #ax[1].plot(x2, np.real(Psi2[:, n2]))
    #ax[1].plot(x2, np.imag(Psi2[:, n2]))
    ax[2].plot(x2, rhob013)
    #ax[2].plot(x2, Vb01)
    #ax[2].plot(x2, np.real(Psi2[:, n3]))
    #ax[2].plot(x2, np.imag(Psi2[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x2, rhob05)
    #ax[0].plot(x2, Vb05)
    #ax[0].plot(x2, np.real(Psib05[:, n1]))
    #ax[0].plot(x2, np.imag(Psib05[:, n1]))
    ax[1].plot(x2, rhob052)
    #ax[1].plot(x2, Vb05)
    #ax[1].plot(x2, np.real(Psib05[:, n2]))
    #ax[1].plot(x2, np.imag(Psib05[:, n2]))
    ax[2].plot(x2, rhob053)
    #ax[2].plot(x2, Vb05)
    #ax[2].plot(x2, np.real(Psib05[:, n3]))
    #ax[2].plot(x2, np.imag(Psib05[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x2, rhob1)
    #ax[0].plot(x2, Vb1)
    #ax[0].plot(x2, np.real(Psib1[:, n1]))
    #ax[0].plot(x2, np.imag(Psib1[:, n1]))
    ax[1].plot(x2, rhob12)
    #ax[1].plot(x2, Vb1)
    #ax[1].plot(x2, np.real(Psib1[:, n2]))
    #ax[1].plot(x2, np.imag(Psib1[:, n2]))
    ax[2].plot(x2, rhob13)
    #ax[2].plot(x2, Vb1)
    #ax[2].plot(x2, np.real(Psib1[:, n3]))
    #ax[2].plot(x2, np.imag(Psib1[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x2b, rho2b)
    #ax[0].plot(x2b, V2b)
    #ax[0].plot(x2b, np.real(Psi2b[:, n1]))
    #ax[0].plot(x2b, np.imag(Psi2b[:, n1]))
    ax[1].plot(x2b, rho2b2)
    #ax[1].plot(x2b, V2b)
    #ax[1].plot(x2b, np.real(Psi2b[:, n2]))
    #ax[1].plot(x2b, np.imag(Psi2b[:, n2]))
    ax[2].plot(x2b, rho2b3)
    #ax[2].plot(x2b, V2b)
    #ax[2].plot(x2b, np.real(Psi2b[:, n3]))
    #ax[2].plot(x2b, np.imag(Psi2b[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    plt.show()

    fig, ax = plt.subplots(2, 1)
    #ax[0].plot(x2b, rho2b)
    # ax[0].plot(x2b, V2b)
    # ax[0].plot(x2b, np.real(Psi2b[:, n1]))
    # ax[0].plot(x2b, np.imag(Psi2b[:, n1]))
    ax[0].plot(x2b, rho2b2)
    ax[1].plot(x2b, rho2b22)
    # ax[1].plot(x2b, V2b)
    # ax[1].plot(x2b, np.real(Psi2b[:, n2]))
    # ax[1].plot(x2b, np.imag(Psi2b[:, n2]))
    #ax[2].plot(x2b, rho2b3)
    # ax[2].plot(x2b, V2b)
    # ax[2].plot(x2b, np.real(Psi2b[:, n3]))
    # ax[2].plot(x2b, np.imag(Psi2b[:, n3]))
    ax[0].set_xlim(0.4, 0.8)
    ax[1].set_xlim(0.4, 0.8)
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    plt.show()

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(x2b, rho2b21)
    #ax[0].plot(x2b, V2b2)
    # ax[0].plot(x2b, np.real(Psi2b1[:, n1]))
    # ax[0].plot(x2b, np.imag(Psi2b1[:, n1]))
    ax[1].plot(x2b, rho2b22)
    #ax[1].plot(x2b, V2b2)
    # ax[1].plot(x2b, np.real(Psi2b22[:, n2]))
    # ax[1].plot(x2b, np.imag(Psi2b22[:, n2]))
    ax[2].plot(x2b, rho2b32)
    #ax[2].plot(x2b, V2b2)
    # ax[2].plot(x2b, np.real(Psi2b32[:, n3]))
    # ax[2].plot(x2b, np.imag(Psi2b32[:, n3]))
    fig.supylabel(r"$\psi$ or $P$")
    fig.supxlabel("x")
    plt.show()

def value_it():

    d = 0.01  # thickness
    v0 = 9.8e5  # height
    k0 = 700
    E = k0**2/2
    T = 1 / (1 + (v0**2 * np.sinh(d * np.sqrt(2*(v0 - E)))**2/(4*E*(v0 - E))))
    print(T)

    File = np.load("arrays.npz")

    Psi = File['arr_0']
    x = File['arr_8']

    size = Psi.shape[1]

    n1 = 1
    n2 = size // 2
    n3 = size - 1

    rho1 = np.conj(Psi[:, n1]) * Psi[:, n1]
    rho2 = np.conj(Psi[:, n2]) * Psi[:, n2]
    rho3 = np.conj(Psi[:, n3]) * Psi[:, n3]

    dx = 5 * 10 ** -4
    sigma = np.sqrt(5e-4)
    print(r"Total Probability: ", np.sum(rho1 * dx, axis=0))
    print(r"Expectation value: ", np.sum(rho1 * dx * x, axis=0))
    print(r"Expectation squared: ", np.sum(rho1 * dx * x ** 2, axis=0))
    print(r"$\sigma^2$: ", np.sum(rho1 * dx * x ** 2, axis=0) - np.sum(rho1 * dx * x, axis=0)**2)

def plot_t():
    d = np.arange(0.001, 0.01, 0.0001)  # thickness
    v0 = 9.8e5  # height
    k0 = 700
    E = k0 ** 2 / 2
    T = 1 / (1 + (v0 ** 2 * np.sinh(d * np.sqrt(2 * (v0 - E))) ** 2 / (4 * E * (v0 - E))))

    plt.plot(d, T)
    plt.xlabel("Barrier Width")
    plt.ylabel("T")
    plt.show()

d = [0.001, 0.005, 0.01]
#for i in range(len(d)):
#quantum_it(0.001)
#plot_it()
value_it()
#plot_t()

