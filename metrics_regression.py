
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

y = np.random.random(size=100) * 10
errors = y**2 * (2 * (np.random.random(size=100)) - 1)
p = y + errors

mse = mean_squared_error(y, p)
mae = mean_absolute_error(y, p)

res = y - p
sns.scatterplot(x=y, y=res)
plt.show()
