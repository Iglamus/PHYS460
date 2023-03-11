import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sc


def fpuit(n_atoms, beta, mode_no):

    # Number of time steps
    run_length = 40000
    # Strength of nonlinearity
    # Mode Amplitude
    mode_amp = 10.0

    V = np.zeros(n_atoms + 2)  # Velocity Vector for the first step
    X = np.zeros((n_atoms + 2, run_length))  # Note we add 2 to n_atoms to account for x_0(t) = x_N+1(t) = 0


    # Frequencies of the normal modes
    omega = 2 * np.sin(np.arange(1, n_atoms + 1) * np.pi / 2 / (n_atoms + 1))
    dt = 200*2*np.pi / run_length / omega[mode_no]
    time = (np.arange(0, run_length) - 1) * dt

    #   Here we use the Euler-Cromer method to populate the first two
    #   states so that we can then use the Verlet method at later times.
    X[0, 0:run_length - 1] = 0
    for k in range(1, n_atoms + 1):
        # FILL IN THE NEXT LINE WITH THE APPROPRIATE INITIAL CONDITION
        X[k, 0] = np.sqrt(2 / (n_atoms + 1)) * mode_amp * np.sin(np.pi * mode_no * k / (n_atoms + 1))

    #   Here we use the Euler-Cromer method to populate the first two
    #   states so that we can then use the Verlet method at later times.
    for k in range(1, n_atoms + 1):
        # Note that initial velocity is 0.
        # In Euler-Cromer, V stepped forward with acceleration at known time.
        V[k] = (X[k + 1, 0] - 2 * X[k, 0] + X[k - 1, 0] + beta * ((X[k + 1, 0] - X[k, 0]) ** 3 - beta * (X[k, 0] - X[k - 1, 0]) ** 3)) * dt

    # In Euler-Cromer, X stepped forward using future time. Vector V is
    # velocity at 2nd time.
    X[1:n_atoms + 1, 1] = X[1:n_atoms + 1, 0] + np.transpose(dt * V[1:n_atoms + 1])

    # Update using the Verlet algorithm
    for l in range(2, run_length):
        for k in range(1, n_atoms + 1):
            # Fill in the next line with the verlet method
            X[k, l] = 2 * X[k, l - 1] - X[k, l - 2] + (X[k + 1, l - 1] - 2 * X[k, l - 1] + X[k - 1, l - 1] + beta * ((X[k + 1, l - 1] - X[k, l - 1]) ** 3 - beta * (X[k, l - 1] - X[k - 1, l - 1]) ** 3)) * dt ** 2

    # Calculate magnitudes of each mode. I recommend using the dst
    # function from scipy
    # FILL IN THE NEXT LINE WITH THE CALCULATION OF THE AMPLITUDE OF EACH
    # MODE
    dstX = np.sqrt(2 / (n_atoms + 1)) * sc.dst(X[1:33], type=1, axis=0, n=n_atoms)

    # Calculate time variation of mode coefficients. This is necessary to
    # calculate the energy in each mode
    dstXprime = np.zeros((n_atoms, run_length))

    # Calculate the time derivative of the mode coefficients
    for i in range(1, run_length - 1):
        dstXprime[:, i] = (dstX[:, i + 1] - dstX[:, i - 1]) / 2 / dt

    # Final time has to use lower order time derivative because time i+1 not
    # available
    dstXprime[:, run_length - 1] = (dstX[:, run_length - 1] - dstX[:, run_length - 2]) / dt

    # Calculate the enrgy in the system
    energy = np.zeros((n_atoms, run_length))
    for i in range(0, n_atoms):
        # FILL IN THE NEXT LINE WITH THE CALCULATION OF THE ENERGY IN THE SYSTEM
        energy[i, :] = 0.5 * (dstXprime[i, :] ** 2 + (omega[i] * dstX[i, :]) ** 2)

    # Calculate epsilon
    eps = omega[mode_no]**2 * dstX[mode_no-1]**2 / n_atoms / 2

    return time, X, dstX, energy, eps


def plotit(t, x, amp, ene, eps, betas):
    # PLOT THE INITIAL AND FINAL DISPLACEMENT OF THE MASSES
    fig, ax = plt.subplots(1, 1)
    temp = x[0]
    ax.plot(temp[:, 0], label="Initial")
    ax.plot(temp[:, -1], label="Final")
    fig.supxlabel('Mass')
    fig.supylabel('Displacement')
    fig.suptitle('Displacements: Initial and Final Times')
    ax.legend(loc=0)
    plt.show()

    # add amplitude vs time
    fig3, ax3 = plt.subplots(len(betas), 1)
    for i in range(len(amp)):
        temp = amp[i]
        ax3[i].plot(t, temp[0], label='Mode 1')
        ax3[i].plot(t, temp[1], label='Mode 2')
        ax3[i].plot(t, temp[2], label='Mode 3')
        ax3[i].set_xlim(0, 100)
    handles, labels = ax3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc='upper right')
    fig3.supxlabel('Time')
    fig3.supylabel('Amplitude')
    plt.show()

    # PLOT THE ENERGIES IN EACH MODE AT THE INITAL AND FINAL TIMES
    fig1, ax1 = plt.subplots(2, 1)
    for i in range(len(ene)):
        temp = ene[i]
        ax1[0].plot(np.arange(1, len(temp) + 1), temp[:, 0], label='Beta = ' + str(betas[i]))
        ax1[1].plot(np.arange(1, len(temp) + 1), temp[:, -1], label='Beta = ' + str(betas[i]))
    fig1.supxlabel('Mode')
    fig1.supylabel('Energy')
    fig1.suptitle('Energies: Initial and Final Times')
    handles, labels = ax1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    plt.show()

    fig4, ax4 = plt.subplots(1, 1)
    temp = ene[1]
    ax4.plot(t, temp[0, :], label='Mode 1')
    ax4.plot(t, temp[1, :], label='Mode 2')
    ax4.plot(t, temp[2, :], label='Mode 3')
    ax4.plot(t, temp[3, :], label='Mode 4')
    ax4.plot(t, temp[4, :], label='Mode 5')
    ax4.plot(t, temp[5, :], label='Mode 6')
    # ax4.set_xlim(400, 600)
    # a.set_ylim(-1, 1)
    ax4.legend(loc=0)
    fig4.supylabel('Energy')
    fig4.supxlabel('Time')
    fig4.suptitle('Energy of First Four Modes')
    plt.show()

    std = []
    for i in range(len(ene)):
        temp = ene[i]
        temp1 = []
        for j in range(len(temp)):
            temp2 = np.std(temp[j])
            temp1.append(temp2)
        std.append(temp1)

    fig2, ax2 = plt.subplots(1, 1)
    for i in range(len(std)):
        ax2.plot(np.arange(1, len(std[i]) + 1), std[i], label='Beta = ' + str(betas[i]))
    fig2.supylabel('Standard Deviation')
    fig2.suptitle('Standard Deviation vs Mode')
    fig2.supxlabel('Mode')
    ax2.legend(loc=0)
    plt.show()

    fig5, ax5 = plt.subplots(len(betas),1)
    for i in range(len(eps)):
        ax5[i].plot(t, eps[i])
    plt.show()

    epsc = [250, 112, 82, 50, 35, 20, 15]
    n = [2, 3, 4, 5, 6, 7, 8]
    plt.plot(n, epsc)
    plt.xlabel('N')
    plt.ylabel('Critical Epsilon')
    plt.show()

if __name__ == "__main__":

    betas = [0, 0.3, 1]
    x, d, e, eps = [], [], [], []
    for beta in betas:
        t, temp2, temp3, temp4, temp5 = fpuit(32, beta, 1)
        x.append(temp2)
        d.append(temp3)
        e.append(temp4)
        eps.append(temp5)
    plotit(t, x, d, e, eps, betas)
