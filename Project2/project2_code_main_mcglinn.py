import numpy as np

import chaos_balls as cb
import matplotlib.pyplot as plt
import winsound


def norm(m1, m2, x1, v1, x2, v2):
    E = 0.5 * m1 * v1**2 + 0.5* m2 * v2**2 + 9.8*m1*x1 + 9.8*m2*x2
    m = m1 + m2
    nx1 = m * 9.8 * x1 / E
    nx2 = m * 9.8 * x2 / E
    nv1 = v1 * np.sqrt(m / E)
    nv2 = v2 * np.sqrt(m / E)

    return m1, m2, nx1, nv1, nx2, nv2


def plotit(time, x1, x2, v1, v2, x2ps, v2ps, mass, x2ps9, v2ps9):

    NCorr = []
    fig2, ax2 = plt.subplots(3, 1)

    for i in range(len(mass)):
        fig, ax = plt.subplots(2, 1, figsize=(11, 11))
        ax[1].plot(time, v2[i], label='Ball 2')
        ax[1].plot(time, v1[i], label='Ball 1')
        ax[1].set_ylabel('V')
        ax[1].yaxis.set_ticks_position('both')
        ax[1].xaxis.set_ticks_position('both')

        ax[0].plot(time, x2[i], label='Ball 2')
        ax[0].plot(time, x1[i], label='Ball 1')
        ax[0].set_ylabel('Position')
        ax[0].yaxis.set_ticks_position('both')
        ax[0].xaxis.set_ticks_position('both')

        for item in ax:
            item.set_xlabel('Time')
            item.set_xlim(0, 20)
            item.legend(loc=0, fontsize=10)
        fig.savefig('x_v' + str(mass[i]) + '.png')

        fig1, ax1 = plt.subplots(1, 1)
        ax1.plot(x2ps[i], v2ps[i], '.')
        ax1.set_xlabel(r'$x_2$')
        ax1.set_ylabel(r'$v_2$', rotation=0)
        ax1.yaxis.set_ticks_position('both')
        ax1.xaxis.set_ticks_position('both')
        fig1.savefig('Poincare' + str(mass[i]) + '.png')

        ax2[i].plot(time, x2[i])
        ax2[i].plot(time, x1[i])
        ax2[i].set_xlim(0, 20)

        #tnew, lag, tempNCorr = corr(x2[i], time)
        #NCorr.append(tempNCorr)
    fig2.supylabel('Position')
    fig2.supxlabel('Time')
    fig2.savefig('x.png')

    color = ['r', 'b', 'y', 'g', 'c', 'm']

    fig1, ax1 = plt.subplots(1, 1)
    for i in range(len(v2ps9)):
        ax1.plot(x2ps9[i], v2ps9[i], '.', c=color[i])
    ax1.set_xlabel(r'$x_2$')
    ax1.set_ylabel(r'$v_2$', rotation=0)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    fig1.savefig('Poincare.png')

    #fig2, ax2 = plt.subplots(3, 1)
    #ax2[0].plot(tnew[:round(len(tnew)/2)], NCorr[0])
    #ax2[1].plot(tnew[:round(len(tnew)/2)], NCorr[1])
    #ax2[2].plot(tnew[:round(len(tnew)/2)], NCorr[2])
    #for item in ax2:
    #    item.set_xlim(0,100)
    #    #item.set_ylim(-1, 1)
    #fig2.supylabel(r'$A(t)$', rotation=0)
    #fig2.supxlabel('Time (s)')
    #fig2.savefig('Corr.png')

    #fig3, ax3 = plt.subplots(1, 1)
    #ax3.plot(tnew[:round(len(tnew)/2)], NCorr[2])
    #ax3.set_xlabel('Time (s)')
    #ax3.set_ylabel(r'$A(t)$', rotation=0)
    #ax3.set_xlim(0,200)
    #fig3.savefig('Corr3.png')


def corr(x2, time):
    lag = 100
    x2new = np.zeros(round(len(x2)/lag))
    tnew = np.zeros(round(len(time)/lag))
    for i in range(len(x2new)):
        x2new[i] = x2[i*lag]
        tnew[i] = time[i*lag]
    xp = x2new - np.average(x2new)
    lagmax = round(len(x2new) / 2)
    Corr = np.zeros(lagmax)
    for i in range(lagmax):
        Corr[i] = sum(np.multiply(xp[1:(len(x2new)-i)], xp[i+1:len(x2new)]))/(len(x2new)-i)
    NCorr = Corr / np.std(x2)**2

    return tnew, lag, NCorr


def main():
    mass = [0.5, 1, 9]
    x1cb = np.zeros((len(mass), 2**22))
    x2cb = np.zeros((len(mass), 2**22))
    v1cb = np.zeros((len(mass), 2**22))
    v2cb = np.zeros((len(mass), 2**22))
    x2ps = []
    v2ps = []
    v2ps9 = []
    x2ps9 = []
    for i in range(len(mass)):
        m1, m2, x1, v1, x2, v2 = norm(1, mass[i], 1, 0, 3, 0)
        time, x1cb[i], x2cb[i], v1cb[i], v2cb[i], tempx2ps, tempv2ps = cb.chaos_balls(m1, m2, x1, v1, x2, v2)
        x2ps.append(tempx2ps)
        v2ps.append(tempv2ps)
    v2ps9.append(tempv2ps)
    x2ps9.append(tempx2ps)
    m1, m2, x1, v1, x2, v2 = norm(1, mass[-1], 1, 0, 2, 0)
    _, _, _, _, _, tempx2ps, tempv2ps = cb.chaos_balls(m1, m2, x1, v1, x2, v2)
    v2ps9.append(tempv2ps)
    x2ps9.append(tempx2ps)
    m1, m2, x1, v1, x2, v2 = norm(1, mass[-1], 1, 0, 1.1, 0)
    _, _, _, _, _, tempx2ps, tempv2ps = cb.chaos_balls(m1, m2, x1, v1, x2, v2)
    v2ps9.append(tempv2ps)
    x2ps9.append(tempx2ps)
    plotit(time, x1cb, x2cb, v1cb, v2cb, x2ps, v2ps, mass, x2ps9, v2ps9)

    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


if __name__ == "__main__":
    main()

