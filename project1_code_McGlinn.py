"""
2/26/2018: Project 1 starter code
"""
import numpy as np
import matplotlib.pyplot as plt
from math import exp


def radioactive_decay_a(N_0, tau, dt):
    '''
    Program calculates number of nuclei a N(t) as a function of time by approximating
    the solution of differential equation of radioactive decay via the Euler method

   Declare and explain or variables explicitly:
   Input:
       N_0    = initial number of nuclei
       tau    = time constant
       dt     = time step
    '''
    # Calculate the amount of steps needed
    time_steps = int(5 * tau / dt)

    # Initialize the numerical and analytical solutions
    N = np.zeros(time_steps)
    exact_N = np.zeros(time_steps)

    # Create a time vector
    time = np.zeros(time_steps)

    # Establish the initial conditions
    N[0] = N_0
    exact_N[0] = N_0

    for i in range(time_steps - 1):
        N[i + 1] = N[i] - (N[i] / tau) * dt
        time[i + 1] = time[i] + dt
        exact_N[i + 1] = N_0 * exp(-time[i + 1] / tau)

    plt.figure()
    plt.plot(time, N)
    plt.title('Number of Radioactive Nuclei vs. Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Nuclei')
    plt.legend([r'$N_a(t)$', r'$exact N_a(t)$'])
    plt.savefig('Nuclei_A_' + str(dt) + '.png', dpi=500)

def radioactive_decay_ab(N_0a, tau_a, N_0b, tau_b, dt):
    '''
    Program calculates number of nuclei N(t) as a function of time by approximating
    the solution of differential equation of radioactive decay via the Euler method

   Declare and explain or variables explicitly:
   Input:
       N_0a    = initial number of nuclei a
       tau_a   = time constant
       N_0b    = initial number of nuclei b
       ratio   = ratio of tau_a/tau_b
       dt      = time step
    '''
    # Calculate the amount of steps needed
    time_steps = int(5 * tau_a / dt)

    # Initialize the numerical and analytical solutions
    N_a = np.zeros(time_steps)
    exact_N_a = np.zeros(time_steps)
    N_b = np.zeros(shape=(len(tau_b), time_steps))
    exact_N_b = np.zeros(shape=(len(tau_b), time_steps))

    # Create a time vector
    time = np.zeros(time_steps)

    # Establish the initial conditions
    N_a[0] = N_0a
    exact_N_a[0] = N_0a

    for j in range(len(tau_b)):
        N_b[j][0] = N_0b
        exact_N_b[j][0] = N_0b
        for i in range(time_steps - 1):
            N_a[i + 1] = N_a[i] - (N_a[i] / tau_a) * dt
            time[i + 1] = time[i] + dt
            exact_N_a[i + 1] = N_0a * exp(-time[i + 1] / tau_a)
            N_b[j][i+1] = N_b[j][i] + (N_a[i] / tau_a) * dt - (N_b[j][i] * tau_a / tau_b[j]) * dt
            if tau_b[j] == tau_a:
                exact_N_b[j][i+1] = N_0b*exp(-time[i+1]/tau_a) + N_0a*time[i+1]/tau_a * exp(-time[i+1]/tau_a)
            else:
                exact_N_b[j][i+1] = (N_0b - (N_0a*tau_b[j])/(tau_a - tau_b[j])) * exp(-time[i + 1] / tau_b[j]) + exp(-time[i + 1] / tau_a) * (N_0a*tau_b[j])/(tau_a - tau_b[j])

    return N_a, N_b, exact_N_a, exact_N_b, time

def plotfigs(N_a, N_b, dt, tau_b, time):

    for i in range(len(N_b)):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.plot(time, N_a, label='Nuclei A')
        # ax.plot(time, exact_N_a, label='Exact')
        ax.plot(time, N_b[i], label='Nuclei B')
        ax.set_title('Number of Radioactive Nuclei A and B vs. Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Nuclei')
        ax.legend(loc="upper right")

        fig.savefig('Nuclei_AB_' + str(tau_b[i]) + '_' + str(dt) + '.png', dpi=500)

    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.set_title('Number of Radioactive Nuclei B vs. Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Nuclei')

    for i in range(len(N_b)):
        ax.plot(time, N_b[i], label=r'$\tau$' + ' = ' + str(tau_b[i]))
        # ax.plot(time, N_a, label='Nuclei A')
        # ax.plot(time, exact_N_a, label='Exact')

    ax.legend(loc="upper right")

    fig.savefig('Nuclei_B_' + str(dt) + '.png', dpi=500)

    plt.close(fig)


'''
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].set_title('Number of Radioactive Nuclei B vs. Time, ' + r'$\tau = $' + str(tau_b[1]))
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Number of Nuclei')

    ax[0].plot(time, N_b[0], label='Euler')
    ax[0].plot(time, exact_N_b[0], label='Exact')
    # ax.plot(time, exact_N_a, label='Exact')

    ax[0].legend(loc="upper right")

    fig.savefig('Nuclei_B_comp' + str(dt) + '.png', dpi=500)
    '''


radioactive_decay_a(200, 1, 0.01)
Na1, Nb1, eNa1, eNb1, time = radioactive_decay_ab(200, 1, 10, [0.5, 1, 1.5], 0.01)
Na2, Nb2, eNa2, eNb2, time2= radioactive_decay_ab(200, 1, 10, [0.5, 1, 1.5], 0.0001)
Na3, Nb3, eNa3, eNb3, time3 = radioactive_decay_ab(200, 1, 10, [0.5, 1, 1.5], 0.1)
plotfigs(Na1, Nb1, 0.01, [0.5, 1, 1.5], time)

fig, ax = plt.subplots(1, 3, figsize=(12, 6), sharey=True)

fig.suptitle('Number of Radioactive Nuclei B vs. Time')

#ax[0].plot(time, Na1, label='Nuclei A')
# ax.plot(time, exact_N_a, label='Exact')
ax[0].plot(time, Nb1[1], label='Numeric', ls='--')
ax[0].plot(time, eNb1[1], label='Analytical')
ax[0].set_title(r'$\Delta t = 0.01$')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Number of Nuclei')
ax[0].legend(loc="upper right")
ax[0].set_ylim(0, 85)
ax[0].set_xlim(0, 3)

ax[1].plot(time2, Nb2[1], label='Numeric', ls='--')
ax[1].plot(time2, eNb2[1], label='Analytical')
ax[1].set_title(r'$\Delta t = 0.0001$')
ax[1].set_xlabel('Time')
#ax[1].set_ylabel('Number of Nuclei')
ax[1].legend(loc="upper right")
ax[1].set_xlim(0, 3)

ax[2].plot(time3, Nb3[1], label='Numeric', ls='--')
ax[2].plot(time3, eNb3[1], label='Analytical')
ax[2].set_title(r'$\Delta t = 0.1$')
ax[2].set_xlabel('Time')
#ax[2].set_ylabel('Number of Nuclei')
ax[2].legend(loc="upper right")
ax[2].set_xlim(0, 3)

plt.tight_layout()

fig.savefig('Nuclei_AB_comp' + '.png', dpi=500)
