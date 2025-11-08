from CMeans import FuzzyCMeans
from utils import load_dataset_from_csv, export_jc_iterations_to_excel, export_cluster_file, plot_final_clusters
import numpy as np

def calculate_rc_ratios(c_values, jc_values):
    ratios = {}
    jc_dict = dict(zip(c_values, jc_values))
    
    for c in range(3, 10):  
        if c-1 in jc_dict and c in jc_dict and c+1 in jc_dict:
            numerator = abs(jc_dict[c] - jc_dict[c+1])
            denominator = abs(jc_dict[c-1] - jc_dict[c])
            ratios[c] = numerator / denominator if denominator != 0 else np.inf
        else:
            ratios[c] = np.inf
    
    return ratios

def main_development():
    """Run on Data Set 5 for development"""
    print("=" * 60)
    print("DEVELOPMENT PHASE - Data Set 5")
    print("=" * 60)
    
    train_data, test_data = load_dataset_from_csv("data/Dataset-5.csv", train_ratio=0.8)
    print(f"Training data: {train_data.shape}, Testing data: {test_data.shape}")
    
    c_min, c_max = 2, 10
    c_values = list(range(c_min, c_max+1))
    # jc_values = []
    # iterations_list = []
    #
    # print("\nTesting cluster counts from 2 to 10:")
    # for c in c_values:
    #     print(f"  c = {c}...", end=" ")
    #     cmeans = FuzzyCMeans(train_data, c=c, m=2, epsilon=0.001)
    #     centroids, obj_func, centroid_ids, iterations_count = cmeans.train(c)
    #     jc_values.append(obj_func)
    #     iterations_list.append(iterations_count)
    #     print(f"Jc = {obj_func:.2f}, Iterations = {iterations_count}")

    cmeans = FuzzyCMeans(train_data, m=2, epsilon=0.001)
    optimal_c, jc_values, ratios, iterations_list = cmeans.optimize_c_value(c_min, c_max)
    jc_values = [jc_values[c] for c in range(c_min, c_max)]
    iterations_list = [iterations_list[c] for c in range(c_min, c_max)]

    print(f"\nRc Ratios (c from 3 to 9):")
    for c in range(c_min+1, c_max):
        print(f"  c={c}: Rc = {ratios[c]:.4f}")
    
    # Find optimal c (minimum Rc ratio)
    print(f"\nOptimal c: {optimal_c} (minimum Rc = {ratios[optimal_c]:.4f})")
    
    # Train final model with optimal c
    print(f"\nTraining final model with optimal c = {optimal_c}")
    final_model = FuzzyCMeans(train_data, c=optimal_c, m=2, epsilon=0.001)
    cluster_centroids, jc_value, cluster_ids, iters = final_model.train()
    
    # Export Jc vs Iterations to Excel
    export_jc_iterations_to_excel(c_values, jc_values, iterations_list, "jc_iterations_dev.xlsx")
    
    # Create file C with cluster assignments 
    export_cluster_file(train_data, cluster_ids, "C_output_dev.csv")
    
    # Plot final clusters
    plot_final_clusters(train_data, cluster_ids, final_model.c, "Data Set 5 - Final Clusters (Training Data)")
    
    # Save centroids for classification
    np.savetxt("centroids_dev.csv", cluster_centroids, delimiter=',')
    print("Centroids saved to centroids_dev.csv")
    
    # CLASSIFICATION 
    print("\n" + "=" * 50)
    print("CLASSIFICATION  - Testing Data")
    print("=" * 50)
    
    test_cluster_ids, test_jc = final_model.test(test_data)
    export_cluster_file(test_data, test_cluster_ids, "C_test_output_dev.csv")
    plot_final_clusters(test_data, test_cluster_ids, final_model.c, "Data Set 5 - Classified Test Data")

    print("Classification completed successfully!")
    return final_model

def main_submission():
    """Run on Data Set 2 for final submission"""
    print("\n" + "=" * 60)
    print("SUBMISSION PHASE - Data Set 2")
    print("=" * 60)
    
    # Load Data Set 2 
    train_data, test_data = load_dataset_from_csv("data/Dataset-2.csv", train_ratio=0.8)
    print(f"Training data: {train_data.shape} (480 points)")
    print(f"Testing data: {test_data.shape} (140 points)")
    
    print("\nFinding optimal c using Rc ratio...")
    temp_model = FuzzyCMeans(train_data, c=None, m=2, epsilon=0.001)
    optimal_c = temp_model.c
    print(f"Optimal c found: {optimal_c}")
    
    # Generate data for plotting (c from 2 to 10)
    c_values = list(range(2, 11))
    jc_values = []
    iterations_list = []
    
    for c in c_values:
        model = FuzzyCMeans(train_data, c=c, m=2, epsilon=0.001)
        jc_values.append(model.obj_func)
        iterations_list.append(model.iterations_count)
    
    export_jc_iterations_to_excel(c_values, jc_values, iterations_list, "jc_iterations_submission.xlsx")
    
    final_model = FuzzyCMeans(train_data, c=optimal_c, m=2, epsilon=0.001)
    
    # CLASSIFICATION of 140 test points 
    print("\nClassifying 140 test points...")
    test_cluster_ids, test_jc, test_U = final_model.test(test_data)
    
    # Export cluster assignments for 140 test points
    export_cluster_file(test_cluster_ids, test_data, "C_output_submission.csv")
    
    # Plot final clusters for 140 test points 
    plot_final_clusters(test_data, test_cluster_ids, "Data Set 2 - Final Clusters (140 Test Points)")    
    np.savetxt("centroids_submission.csv", final_model.cluster_centroids, delimiter=',')
    
    print("\n" + "=" * 50)
    print("SUBMISSION FILES GENERATED:")
    print("1. jc_iterations_submission.xlsx - Excel file with chart")
    print("2. final_cluster_plot.png - Cluster plot (140 test points)") 
    print("3. C_output_submission.csv - Cluster assignments")
    print("4. centroids_submission.csv - Cluster centroids")
    print("=" * 50)

if __name__ == "__main__":
    main_development()
    # main_submission()