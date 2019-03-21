import Orange
import matplotlib.pyplot as plt

names = ["auto-band", "auto-weka", "auto-sklearn"]
avranks = [1.76316, 1.94737, 2.28947]

cd = Orange.evaluation.scoring.compute_CD(avranks, 19, alpha="0.05", type="nemenyi")#tested on 12 datasets
Orange.evaluation.scoring.graph_ranks('C:\Users\silvia\Desktop\\cd2.png',avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()