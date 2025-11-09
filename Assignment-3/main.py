from CMeans import FuzzyCMeans
from utils import (
    load_dataset_from_csv,
    plot_jc_iterations,
    export_cluster_file,
    plot_final_clusters,
    plot_jc_iterations
)
import numpy as np


def main_development():
    """Run on Data Set 5 for development"""
    print("=" * 60)
    print("DEVELOPMENT PHASE - Data Set 5")
    print("=" * 60)

    train_data, test_data = load_dataset_from_csv("data/Dataset-5.csv", train_ratio=0.8)
    print(f"Training data: {train_data.shape}, Testing data: {test_data.shape}")

    c_min, c_max = 2, 10
    cmeans = FuzzyCMeans(train_data, c=None, m=2, epsilon=0.001, auto_train=False)
    optimal_c, jc_values_dict, ratios_dict, iterations_dict = cmeans.optimize_c_value(c_min, c_max)

    print(f"\nOptimal c: {optimal_c} (minimum Rc = {ratios_dict[optimal_c]:.4f})")

    print(f"\nRc Ratios (c from 3 to 9):")
    for c in range(3, 10):
        if c in ratios_dict:
            print(f"  c={c}: Rc = {ratios_dict[c]:.4f}")

    c_values = list(range(c_min, c_max + 1))
    jc_values = [jc_values_dict[c] for c in c_values]
    iterations_list = [iterations_dict[c] for c in c_values]

    final_model = FuzzyCMeans(train_data, c=optimal_c, m=2, epsilon=0.001, auto_train=True)

    plot_jc_iterations(c_values, jc_values, iterations_list, "jc_iterations_dev.xlsx")

    # ✅ New addition: Save Jc vs Iterations plot
    plot_jc_iterations(c_values, jc_values, iterations_list, "Data Set 5")

    export_cluster_file(train_data, final_model.centroid_classification, "C_output_dev.csv")

    plot_final_clusters(
        train_data,
        final_model.centroid_classification,
        final_model.c,
        centroids=final_model.cluster_centroids,
        title="Data Set 5 - Final Clusters (Training Data)"
    )

    np.savetxt(
        "centroids_dev.csv",
        final_model.cluster_centroids,
        delimiter=",",
        fmt="%.6f",
        header="x,y",
        comments=""
    )
    print("Centroids saved to centroids_dev.csv")

    print("\n" + "=" * 50)
    print("CLASSIFICATION  - Testing Data")
    print("=" * 50)

    test_cluster_ids, test_jc = final_model.test(test_data)
    export_cluster_file(test_data, test_cluster_ids, "C_test_output_dev.csv")

    plot_final_clusters(
        test_data,
        test_cluster_ids,
        final_model.c,
        centroids=final_model.cluster_centroids,
        title="Data Set 5 - Classified Test Data"
    )

    print("Classification completed successfully!")
    return final_model


def main_submission():
    """Run on Data Set 2 for final submission"""
    print("\n" + "=" * 60)
    print("SUBMISSION PHASE - Data Set 2")
    print("=" * 60)

    train_data, test_data = load_dataset_from_csv("data/Dataset-2.csv", train_ratio=0.8)
    print(f"Training data: {train_data.shape} (480 points)")
    print(f"Testing data: {test_data.shape} (140 points)")

    print("\nFinding optimal c using Rc ratio...")
    c_min, c_max = 2, 10
    cmeans_optimizer = FuzzyCMeans(train_data, c=None, m=2, epsilon=0.001, auto_train=False)
    optimal_c, jc_values_dict, ratios_dict, iterations_dict = cmeans_optimizer.optimize_c_value(c_min, c_max)

    print(f"Optimal c: {optimal_c} (minimum Rc = {ratios_dict[optimal_c]:.4f})")

    c_values = list(range(c_min, c_max + 1))
    jc_values = [jc_values_dict[c] for c in c_values]
    iterations_list = [iterations_dict[c] for c in c_values]

    print(f"\nRc Ratios (c from 3 to 9):")
    for c in range(3, 10):
        if c in ratios_dict:
            print(f"  c={c}: Rc = {ratios_dict[c]:.4f}")

    export_jc_iterations_to_excel(c_values, jc_values, iterations_list, "jc_iterations_submission.xlsx")

    # ✅ New addition: Save Jc vs Iterations plot for submission
    plot_jc_iterations(c_values, jc_values, iterations_list, "Data Set 2")

    print(f"\nUsing optimal model with c = {optimal_c}")
    final_model = FuzzyCMeans(train_data, c=optimal_c, m=2, epsilon=0.001, auto_train=True)

    print("\nClassifying 140 test points...")
    test_cluster_ids, test_jc = final_model.test(test_data)

    export_cluster_file(test_data, test_cluster_ids, "C_output_submission.csv")

    plot_final_clusters(
        test_data,
        test_cluster_ids,
        final_model.c,
        centroids=final_model.cluster_centroids,
        title="Data Set 2 - Final Clusters (140 Test Points)"
    )

    np.savetxt(
        "centroids_submission.csv",
        final_model.cluster_centroids,
        delimiter=",",
        fmt="%.6f",
        header="x,y",
        comments=""
    )

    print("\n" + "=" * 50)
    print("SUBMISSION FILES GENERATED:")
    print("1. jc_iterations_submission.xlsx - Excel file with chart")
    print("2. Data_Set_2_Jc_vs_Iterations.png - Objective Function plot")
    print("3. final_cluster_plot.png - Cluster plot (140 test points)")
    print("4. C_output_submission.csv - Cluster assignments")
    print("5. centroids_submission.csv - Cluster centroids")
    print("=" * 50)


if __name__ == "__main__":
    main_development()
    # main_submission()
