import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.styles import Font, Alignment


def load_dataset_from_csv(file_path: str, train_ratio: float = None, test_ratio: float = None,
                          train_samples: int = None, test_samples: int = None):
    """
    Load dataset from CSV file and split into train/test based on user input.

    Priority:
      1. If user specifies sample counts → use them.
      2. Else if user specifies percentages → use them.
      3. Else → default split 75% train / 25% test.
    """
    df = pd.read_csv(file_path)
    data = np.array(df)
    total_samples = len(data)

    # Remove empty columns
    i = 0
    while i < data.shape[1]:
        if np.isnan(data[:, i]).all():
            data = np.delete(data, i, axis=1)
        else:
            i += 1

    # Handle sample-based splitting
    if train_samples or test_samples:
        if train_samples:
            test_samples = total_samples - train_samples
        else:
            train_samples = total_samples - test_samples

        train_data = data[:train_samples]
        test_data = data[train_samples:train_samples + test_samples]

    # Handle ratio-based splitting
    elif train_ratio or test_ratio:
        if train_ratio:
            test_ratio = 1 - train_ratio
        else:
            train_ratio = 1 - test_ratio
        train_count = int(train_ratio * total_samples)
        train_data = data[:train_count]
        test_data = data[train_count:]

    # Default split 75/25
    else:
        train_count = int(0.75 * total_samples)
        train_data = data[:train_count]
        test_data = data[train_count:]

    return train_data, test_data


def export_jc_iterations_to_excel(c_values, jc_values, iterations, filename="jc_iterations.xlsx"):
    """
    Exports the number of clusters (c), Objective Function (Jc),
    and Iterations count to an Excel file and embeds a dual-axis line chart.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Jc_vs_Iterations"

    headers = ["Clusters (c)", "Objective Function (Jc)", "Iterations"]
    ws.append(headers)

    header_font = Font(bold=True)
    for col in range(1, len(headers) + 1):
        ws.cell(row=1, column=col).font = header_font
        ws.cell(row=1, column=col).alignment = Alignment(horizontal='center')

    for c, jc, iters in zip(c_values, jc_values, iterations):
        ws.append([c, jc, iters])

    chart = LineChart()
    chart.title = "Data Set 5"
    chart.style = 10
    chart.y_axis.title = "Objective Function (Jc)"
    chart.x_axis.title = "Number of Clusters (c)"
    chart.x_axis.crosses = "min"
    chart.legend.position = "b"

    data_ref = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=len(c_values) + 1)
    cats_ref = Reference(ws, min_col=1, min_row=2, max_row=len(c_values) + 1)
    chart.add_data(data_ref, titles_from_data=True)
    chart.set_categories(cats_ref)

    jc_chart = LineChart()
    it_chart = LineChart()
    jc_chart.add_data(Reference(ws, min_col=2, min_row=1, max_row=len(c_values) + 1), titles_from_data=True)
    it_chart.add_data(Reference(ws, min_col=3, min_row=1, max_row=len(c_values) + 1), titles_from_data=True)
    jc_chart.set_categories(cats_ref)
    it_chart.set_categories(cats_ref)

    jc_chart.y_axis.title = "Objective Function (Jc)"
    jc_chart.x_axis.title = "Number of Clusters (c)"
    it_chart.y_axis.axId = 200 
    it_chart.y_axis.title = "Number of Iterations"
    it_chart.y_axis.crosses = "max"

    jc_chart += it_chart
    jc_chart.title = "Data Set 5"
    jc_chart.legend.position = "b"

    ws.add_chart(jc_chart, "E2")

    wb.save(filename)
    print(f"[+] Exported Excel file with embedded chart → '{filename}' successfully!")


def export_cluster_file(U_matrix, data, filename="C_output.csv"):
    """
    Creates a file with 3 columns:
    x, y, cluster_id  (cluster_id from 0 to c-1)
    """

    cluster_ids = np.argmax(U_matrix, axis=0)

    x = data[:, 0]
    y = data[:, 1]
    rows = zip(x, y, cluster_ids)

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "Cluster_ID"])
        writer.writerows(rows)

    print(f"[+] Exported cluster assignments to '{filename}' successfully!")


def plot_final_clusters(train_data, U_matrix, title="Final Cluster Illustration"):
    """
    Plots the clustered data points in 2D, where each cluster is shown in a unique color.
    """
    cluster_ids = np.argmax(U_matrix, axis=0)
    num_clusters = U_matrix.shape[0]

    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_clusters))

    for k in range(num_clusters):
        cluster_points = train_data[cluster_ids == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[k], label=f"Cluster {k+1}", s=30)

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)

    output_path = "final_cluster_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved cluster plot to '{output_path}' successfully!")

    plt.show()
