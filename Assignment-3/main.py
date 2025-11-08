""" Main File. This is where all the implementation will take place including the test and
tuning on Dataset 5 and the final submission run as well"""

from CMeans import FuzzyCMeans
from utils import load_dataset_from_csv, export_jc_iterations_to_excel, plot_final_clusters
import numpy as np

train_data, test_data = load_dataset_from_csv("data/Dataset-5.csv", train_ratio=0.8)

c_values, jc_values, iterations = [], [], []

for c in range(2, 11):
    print(f"Running Fuzzy C-Means for c = {c}")
    cmeans = FuzzyCMeans(np.array(train_data), c)
    jc_values.append(cmeans.obj_func)
    c_values.append(c)
    iterations.append(10) 

export_jc_iterations_to_excel(c_values, jc_values, iterations)
print("\n✅ CSV export complete! Check 'jc_iterations.csv' in your Assignment-3 folder.\n")


best_c = c_values[np.argmin(jc_values)]
print(f"\nBest c identified = {best_c}")

cmeans_final = FuzzyCMeans(np.array(train_data), best_c)

from utils import export_cluster_file
export_cluster_file(cmeans_final.U_matrix, np.array(train_data), filename="C_output.csv")

best_cmeans = FuzzyCMeans(np.array(train_data), c=5) 
plot_final_clusters(np.array(train_data), best_cmeans.U_matrix, title="Dataset 5 - Final Clusters")
