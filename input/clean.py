from scipy.io import loadmat

trainSet = loadmat('matlab.mat')['edges']
with open('Epinions.txt', 'w') as f2w:
    for edge in trainSet:
        u = edge[0] - 1
        v = edge[1] - 1
        sign = edge[2]
        f2w.write(str(u) + '\t' + str(v) + '\t' + str(sign) + '\n')

