#!/usr/bin/env python3
"""
Summarize experiment results from a visualizations folder.
Usage: python scripts/summarize_results.py <viz_folder>
"""

import os
import sys
import glob
import json


def read_json(path):
    with open(path) as f:
        return json.load(f)


def find_file(folder, pattern):
    matches = glob.glob(os.path.join(folder, pattern))
    return matches[0] if matches else None


def summarize(viz_folder):
    experiments = sorted(
        d for d in os.listdir(viz_folder)
        if os.path.isdir(os.path.join(viz_folder, d))
    )

    col_exp = max((len(e) for e in experiments), default=20)
    col_exp = max(col_exp, len("Experiment"))

    header = (
        f"{'Experiment':<{col_exp}}  "
        f"{'AviaNZ Acc':>10}  {'AviaNZ Acc+':>11}  {'AviaNZ F1':>10}  {'AviaNZ Jaccard':>14}  "
        f"{'DOC Acc':>7}  {'DOC Acc+':>8}  {'DOC F1':>7}  {'DOC Jaccard':>11}"
    )
    print(header)
    print("-" * len(header))

    for exp in experiments:
        folder = os.path.join(viz_folder, exp)

        avianz_file = find_file(folder, "*_test_avianz_split_multilabel_report.json")
        doc_file = find_file(folder, "*_test_doc_split_multilabel_report.json")

        vals = {}
        for key, path in [("avianz", avianz_file), ("doc", doc_file)]:
            if path:
                data = read_json(path)
                vals[key] = {
                    "acc": data.get("exact_match_accuracy"),
                    "acc_lab": data.get("exact_match_accuracy_labelled"),
                    "f1": data.get("micro avg", {}).get("f1-score"),
                    "jaccard": data.get("jaccard_score"),
                }
            else:
                vals[key] = {"acc": None, "acc_lab": None, "f1": None, "jaccard": None}

        def fmt(val):
            return f"{val:.4f}" if val is not None else "N/A"

        a, d = vals["avianz"], vals["doc"]
        print(
            f"{exp:<{col_exp}}  "
            f"{fmt(a['acc']):>10}  {fmt(a['acc_lab']):>11}  {fmt(a['f1']):>10}  {fmt(a['jaccard']):>14}  "
            f"{fmt(d['acc']):>7}  {fmt(d['acc_lab']):>8}  {fmt(d['f1']):>7}  {fmt(d['jaccard']):>11}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <viz_folder>")
        sys.exit(1)

    viz_folder = sys.argv[1]
    if not os.path.isdir(viz_folder):
        print(f"Error: '{viz_folder}' is not a directory")
        sys.exit(1)

    summarize(viz_folder)
