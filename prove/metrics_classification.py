
# pip install scikit-plot
import matplotlib.pyplot as plt
import scikitplot as skplt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import numpy as np

def randomize(a, labels, prob=0.2):
    a2 = []
    for el in a:
        if np.random.random() > prob:
            a2.append(el)
        else:
            a2.append( np.random.choice(labels) )
    return a2

labels = ['cronaca', 'politica', 'sport']
y = np.random.choice(labels, 1000)
p = randomize(y, labels)

# Accuracy, misclassification, precision, recall
acc = accuracy_score(y, p)
print(f'Accuracy {acc}')
print(f'Misclassification {1 - acc}')

# (fare le altre metriche a manina???)
# SPEZZARE IN DUE VIDEO???

report = classification_report(y, p)
print(report)

skplt.metrics.plot_confusion_matrix(y, p)
plt.show()