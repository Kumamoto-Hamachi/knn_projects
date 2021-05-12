def divide_errors(y_pred, y_test):
    wrong_indices = {}
    for i, (pred, ans) in enumerate(zip(y_pred, y_test)):
        if pred != ans:
            wrong_indices.setdefault(f"{ans}-{pred}", [])
            wrong_indices[f"{ans}-{pred}"].append(i)
    return wrong_indices


def count_pred_labels(y):  # count each row num in heatmap
    pred_labels = [0] * 10  # index express a true label
    for ans in y:
        pred_labels[int(ans)] += 1
    print("pred_labels", pred_labels)  # debug
    return pred_labels


def identify_frequent_combinations(y_pred, y_test, threshold_rate):
    wrong_indices = divide_errors(y_pred, y_test)
    pred_labels = count_pred_labels(y_test)
    frequent_combs = {}
    for com, indices in wrong_indices.items():
        ans = int(com[0])
        rate = len(indices) / pred_labels[ans]
        if rate >= threshold_rate:
            frequent_combs[com] = wrong_indices[com]
    return frequent_combs
