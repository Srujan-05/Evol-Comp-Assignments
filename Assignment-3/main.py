from CMeans import FuzzyCMeans
from utils import load_dataset_from_csv, plot_jc_iterations, export_cluster_file, plot_final_clusters, load_interspersed_data
import numpy as np

def main_development():
    """Run on Data Set 5 for development"""
    print("=" * 60)
    print("DEVELOPMENT PHASE - Data Set 5")
    print("=" * 60)
    
    train_data, test_data = load_interspersed_data("data/Dataset-5.csv")
    print(f"Training data: {train_data.shape}, Testing data: {test_data.shape}")
    
    c_min, c_max = 2, 10
    cmeans = FuzzyCMeans(train_data, c=None, m=2, epsilon=0.001, auto_train=False)
    optimal_c, jc_values_dict, ratios_dict, iterations_dict = cmeans.optimize_c_value(c_min, c_max)

    print(f"\nOptimal c: {optimal_c} (minimum Rc = {ratios_dict[optimal_c]:.4f})")

    # Print Rc ratios 
    print(f"\nRc Ratios (c from 3 to 9):")
    for c in range(3, 10):
        if c in ratios_dict:
            print(f"  c={c}: Rc = {ratios_dict[c]:.4f}")

    c_values = list(range(c_min, c_max + 1))
    jc_values = [jc_values_dict[c] for c in c_values]
    iterations_list = [iterations_dict[c] for c in c_values]

    # Train final model with optimal c
    final_model = FuzzyCMeans(train_data, c=optimal_c, m=2, epsilon=0.001, auto_train=True)
    
    plot_jc_iterations(c_values, jc_values, iterations_list, "jc_iterations_dev")
    export_cluster_file(train_data, final_model.centroid_classification, "C_output_dev.csv")

    # Changes made: now explicitly passes centroids and title
    plot_final_clusters(train_data, final_model.centroid_classification, final_model.c, centroids=final_model.cluster_centroids, title="Data Set 5 - Final Clusters (Training Data)")

    # Changes made: Now saves proper centroid coordinates, not cluster assignments
    np.savetxt("centroids_dev.csv", final_model.cluster_centroids, delimiter=",", fmt="%.6f", header="x,y", comments="")
    print("Centroids saved to centroids_dev.csv")
    
    # CLASSIFICATION 
    print("\n" + "=" * 50)
    print("CLASSIFICATION  - Testing Data")
    print("=" * 50)
    
    test_cluster_ids, test_jc = final_model.test(test_data)
    export_cluster_file(test_data, test_cluster_ids, "C_test_output_dev.csv")

    # Changes made: now passes centroids and title correctly again
    plot_final_clusters(test_data, test_cluster_ids, final_model.c, centroids=final_model.cluster_centroids, title="Data Set 5 - Classified Test Data")

    print("Classification completed successfully!")
    return final_model

def main_submission():
    """Run on Data Set 2 for final submission"""
    print("\n" + "=" * 60)
    print("SUBMISSION PHASE - Data Set 2")
    print("=" * 60)
    
    # Load Data Set 2 
    train_data, test_data = load_interspersed_data("data/Dataset-2.csv", data_set_number=2)
    print(f"Training data: {train_data.shape} (480 points)")
    print(f"Testing data: {test_data.shape} (140 points)")
    
    #find optimal c value
    print("\nFinding optimal c using Rc ratio...")
    c_min, c_max = 2, 10
    cmeans_optimizer = FuzzyCMeans(train_data, c=None, m=2, epsilon=0.001, auto_train=False)
    optimal_c, jc_values_dict, ratios_dict, iterations_dict = cmeans_optimizer.optimize_c_value(c_min, c_max)
    
    print(f"Optimal c: {optimal_c} (minimum Rc = {ratios_dict[optimal_c]:.4f})")
    
    c_values = list(range(c_min, c_max + 1))
    jc_values = [jc_values_dict[c] for c in c_values]
    iterations_list = [iterations_dict[c] for c in c_values]
    
    # Print Rc ratios and store values in excel
    print(f"\nRc Ratios (c from 3 to 9):")
    for c in range(3, 10):
        if c in ratios_dict:
            print(f"  c={c}: Rc = {ratios_dict[c]:.4f}")
    
    plot_jc_iterations(c_values, jc_values, iterations_list, "jc_iterations_submission")

    #get the final model with optimal c
    print(f"\nUsing optimal model with c = {optimal_c}")
    final_model = FuzzyCMeans(train_data, c=optimal_c, m=2, epsilon=0.001, auto_train = True)
    
    # CLASSIFICATION of 140 test points 
    print("\nClassifying 140 test points...")
    test_cluster_ids, test_jc = final_model.test(test_data)
    
    # Export cluster assignments and plot final clusters for 140 test points
    export_cluster_file(test_data, test_cluster_ids, "C_output_submission.csv")
    plot_final_clusters(test_data, test_cluster_ids, final_model.c, centroids=final_model.cluster_centroids, title="Data Set 2 - Final Clusters (140 Test Points)")
    np.savetxt("centroids_submission.csv", final_model.cluster_centroids, delimiter=',')
    
    print("\n" + "=" * 50)
    print("SUBMISSION FILES GENERATED:")
    print("1. jc_iterations_submission.xlsx - Excel file with chart")
    print("2. final_cluster_plot.png - Cluster plot (140 test points)") 
    print("3. C_output_submission.csv - Cluster assignments")
    print("4. centroids_submission.csv - Cluster centroids")
    print("=" * 50)

if __name__ == "__main__":
    # main_development()
    main_submission()