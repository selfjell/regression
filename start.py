import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error

type_dict = {"brownii": 0,
            "pringlei" : 1,
            "trinervia" : 2,
            "ramosissima" : 3,
            "robusta" : 4,
            "bidentis" : 5}

nitrogen_dict = {'L' : 0, 'M' : 1, 'H' : 2}

def get_and_transform_data():
    csv = pd.read_csv('flaveria.csv').values
    csv = [(nitrogen_dict[x], type_dict[y], z) for (x,y,z) in csv[:,0:3]]

    #Randomization since data is sorted by subspecies so speak
    random.shuffle(csv)

    data = [[x,y] for (x,y,z) in csv]
    targets = [z for (x,y,z) in csv]
    return data, targets



def plot(data, targets, pred):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    for (x,y), z in zip(data,targets):
        if x == 0:
            ax.scatter([x],[y], zs = z, c = 'g')
        elif x == 1:
            ax.scatter([x],[y], zs = z, c = 'y')
        elif x == 2:
            ax.scatter([x],[y], zs = z, c = 'r')
    plt.show()


##DATA IMPORT##
data, targets = get_and_transform_data()

#Splitting data
X_train, y_train = data[:38], targets[:38]
X_test, y_test = data[38:], targets[38:]
#X_train, y_train = data, targets
#X_test, y_test = data, targets
##FIT/PREDICT##
alpha = .1
reg = linear_model.Lasso(alpha = alpha)
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
#print(pred)

##ERROR##
error = [abs(p-y_test[i]) for i, p in enumerate(pred)]
error = sum(error)/len(error)
score = reg.score(X_test, y_test)
print("Alpha: {}".format(alpha))
print("Mean squared error: (best is 0) {}".format(mean_squared_error(y_test, pred)))
print("Own calculated error (want 0): {}".format(error))
print("Variance score (want 1.0): {}".format(score))



#plot(data, targets, pred)


"""
    meshX = [x for [x,y] in data]
    meshY = [y for [x,y] in data]
    xx, yy = np.meshgrid(meshX, meshY)
    zz = np.meshgrid(pred)
    print(zz)
    ax.plot_surface(xx,yy, zz)
    """

"""
csv = get_data()
plot(csv)

brownii = [(x, y) for (x, label, y) in csv[:, 0:3] if label=="brownii"]
pringlei = [(x, y) for (x, label, y) in csv[:, 0:3] if label=="pringlei"]
trinervia = [(x, y) for (x, label, y) in csv[:, 0:3] if label=="trinervia"]
ramosissima = [(x, y) for (x, label, y) in csv[:, 0:3] if label=="ramosissima"]
robusta = [(x, y) for (x, label, y) in csv[:, 0:3] if label=="robusta"]
bidentis = [(x, y) for (x, label, y) in csv[:, 0:3] if label=="bidentis"]

datadict = {"brownii": brownii,
            "pringlei" : pringlei,
            "trinervia" : trinervia,
            "ramosissima" : ramosissima,
            "robusta" : robusta,
            "bidentis" : bidentis}
"""
