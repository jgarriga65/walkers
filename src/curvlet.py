import skWalker as skw
np = skw.np
plt = skw.plt

def getm():
    m = skw.Mission()
    m.levy()
    m.getSize()
    m.T = np.array([(p.x, p.y) for p in m.gTrack])
    m.T -= np.mean(m.T, axis = 0)
    return m

def plotT(m):
    plt.scatter(m.T[:, 0], m.T[:, 1])
    plt.show()

def getWl(m, aX = 180):

    m.W = []
    for w, a, b in zip([1, 2, 4], [0, int(aX/4), int(3*aX/8)], [aX, int(aX/2), int(aX/4)]):
        A = np.arange(np.pi, 0, -np.pi /aX)
        B = np.array([np.cos(w *A), np.sin((w *A -np.pi/2))]).transpose()
        W = np.zeros(aX *2).reshape(aX, 2)
        W[a:a+b, :] = B[0:b, :]
        m.W.append(W)

    for w in m.W: plt.plot(w)
    plt.show()

def getC(m):

    n = m.W[0].shape[0] //2
    m.C = [np.array([np.sum(m.T[i-n:i+n, :] /W, axis = 0) for i in range(m.T.shape[0])]) for W in m.W]


def plotS(m):


    S = [np.mean([C[b, c, :] *m.W *r for b in range(m.C.shape[0])], axis = 1) for c, r in enumerate(m.R)]

    S01 = S[0] + S[1]
    S02 = S01 + S[2] + S[3]
    S03 = S02 + S[4] + S[5] + S[6]

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].scatter(m.T[:, 0], m.T[:, 1], s=0.5)
    axs[0, 1].scatter(S01[:, 0], S01[:, 1], s = 0.5)
    axs[1, 0].scatter(S02[:, 0], S02[:, 1], s = 0.5)
    axs[1, 1].scatter(S03[:, 0], S03[:, 1], s = 0.5)

    plt.show()
