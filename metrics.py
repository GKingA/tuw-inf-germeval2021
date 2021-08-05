import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_heatmap(matrix):
    """
    Draw the heatmap of the matrix.
    """
    plt.style.use("ggplot")
    sns.heatmap(matrix)
    plt.show()


def print_confusion_matrix(matrix, label_dictionary):
    """
    Print out the confusion matrix formatted with labels
    """
    values = "\t".join(label_dictionary.values())
    print(f"*\t{values}")
    for i in label_dictionary:
        row = "\t".join([str(col) for col in matrix[i]])
        print(f"{label_dictionary[i]}\t{row}")


def print_metrics(stats):
    """
    Print out the the Precision, Recall and F1 of the stats
    """
    print("Category\tPrecision\tRecall\tF1")
    for name, stat in stats.items():
        print(f"{name}\t{round(stat['precision']*100, 2)}%\t{round(stat['recall']*100, 2)}%\t"
              f"{round(stat['f1']*100, 2)}%")


def calculate_confusion_matrix(predicted, label, label_dictionary):
    confusion_matrix = np.zeros((len(label_dictionary), len(label_dictionary)))
    for pred, lab in zip(predicted, label):
        for p, l in zip(pred, lab):
            confusion_matrix[p][l] += 1
    return confusion_matrix


def calculate_fscore_from_metrics(stats):
    fscore = lambda x: 0 if x["precision"] + x["recall"] == 0 \
                         else 2 * x["precision"] * x["recall"] / (x["precision"] + x["recall"])
    for _, stat in stats.items():
        stat["precision"] = 0 if stat["tp"] + stat["fp"] == 0 else stat["tp"] / (stat["tp"] + stat["fp"])
        stat["recall"] = 0 if stat["tp"] + stat["fn"] == 0 else stat["tp"] / (stat["tp"] + stat["fn"])
        stat["f1"] = fscore(stat)
    stats["MICRO AVG"] = {
        "precision": sum([stat["tp"] for (name, stat) in stats.items()
                          if name not in ["MICRO AVG", "MACRO AVG"]]) /
                     sum([stat["tp"] + stat["fp"] for (name, stat) in stats.items()
                          if name not in ["MICRO AVG", "MACRO AVG"]]),
        "recall": sum([stat["tp"] for (name, stat) in stats.items()
                       if name not in ["MICRO AVG", "MACRO AVG"]]) /
                  sum([stat["tp"] + stat["fn"] for (name, stat) in stats.items()
                       if name not in ["MICRO AVG", "MACRO AVG"]])}
    stats["MICRO AVG"]["f1"] = fscore(stats["MICRO AVG"])
    stats["MACRO AVG"] = {
        "precision": np.mean([stat["precision"] for (name, stat) in stats.items() if name not in ["MICRO AVG", "MACRO AVG"]]),
        "recall": np.mean([stat["recall"] for (name, stat) in stats.items() if name not in ["MICRO AVG", "MACRO AVG"]])
    }
    stats["MACRO AVG"]["f1"] = fscore(stats["MACRO AVG"])
    return stats


def calculate_metrics_from_confusion_matrix(confusion_matrix, label_dictionary):
    stats = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in label_dictionary.values()}
    for i, row in enumerate(confusion_matrix):
        stats[label_dictionary[i]]['tp'] = row[i]
        stats[label_dictionary[i]]['fp'] = sum(row) - row[i]
        for j, col in enumerate(row):
            if i != j:
                stats[label_dictionary[j]]["fn"] += col
    return stats


def calculate_metrics_from_data(predicted, label, label_dictionary):
    stats = {k: {'tp': 0, 'fp': 0, 'fn': 0} for k in label_dictionary.values()}
    for pred, lab in zip(predicted, label):
        for p, l in zip(pred, lab):
            if p == l:
                stats[label_dictionary][l]['tp'] += 1
            else:
                stats[label_dictionary][p]['fp'] += 1
                stats[label_dictionary][l]['fn'] += 1
    return stats


def calculate_fscore(label_dictionary, **kwargs):
    if "label" in kwargs and "predicted" in kwargs:
        stats = calculate_metrics_from_data(kwargs["predicted"], kwargs["label"], label_dictionary)
    elif "confusion_matrix" in kwargs:
        stats = calculate_metrics_from_confusion_matrix(kwargs["confusion_matrix"], label_dictionary)
    else:
        raise ValueError("The calculate_fscore function expects label and predicted data or "
                         "confusion matrix as parameter."
                         "\nExample: calculate_fscore(label_dictionary, confusion_matrix=confusion_matrix)"
                         "\nor: calculate_fscore(label_dictionary, label=label, predicted=predicted)")
    return calculate_fscore_from_metrics(stats)


def calculate_and_print_metrics(predicted, label, label_dictionary, draw_matrix):
    confusion_matrix = calculate_confusion_matrix(predicted, label, label_dictionary)
    stats = calculate_fscore(label_dictionary, confusion_matrix=confusion_matrix)
    if draw_matrix:
        draw_heatmap(confusion_matrix)
    else:
        print_confusion_matrix(confusion_matrix, label_dictionary)
    print_metrics(stats)
    return confusion_matrix, stats
