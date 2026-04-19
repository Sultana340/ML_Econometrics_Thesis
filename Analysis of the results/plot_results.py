import pandas as pd
df = pd.read_csv("results/agmm/summary_results_abs_0.5_x_image_1000_AGMM.csv")
import matplotlib.pyplot as plt
plt.figure()
plt.bar(df['estimator'], df['avg_MSE'])
plt.xlabel('Estimator')
plt.ylabel('Average MSE')
plt.title('AGMM Performance')
plt.show()