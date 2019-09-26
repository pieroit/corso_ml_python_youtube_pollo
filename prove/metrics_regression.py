from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

y = np.random.random(size=100) * 10
errors = (np.random.random(size=100) * 2) - 1
errors = y/2 + (np.random.random(size=100)*2 - 1)# sistematic error 1
#errors = y**2 * (np.random.random(size=100)*2 - 1)# sistematic error 2
p = y + errors

mse = mean_squared_error(y, p)
mae = mean_absolute_error(y, p)
r2  = r2_score(y, p)

print(f'MSE {mse}, MAE {mae}, R2 {r2}')

# SPEZZARE IL VIDEO IN DUE PARTI?

res = y - p
sns.scatterplot(x=y, y=res)
plt.show()

# ACCENNO AI TOOL COME WHATIF/VISUALIZZAZIONE E AL "DEBUG" IN ML