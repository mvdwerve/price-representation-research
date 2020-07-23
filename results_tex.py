import os
import json
import numpy as np
from convert_cpc import reload_settings

targets = ["movement", "movement_up", "anomaly", "future_anomaly", "positive"]
splits = ["train", "valid", "test"]

# all results
results = []

# iterate over all folders
for folder in os.listdir("logs/"):
    # reload the settings
    settings = reload_settings(os.path.join("logs", folder))

    # the intermediate object
    intermediate = {}

    # iterate over the targets
    for target in targets:
        sets = {}

        for what in splits:
            # load the results
            with open(
                os.path.join("logs", folder, "downstream", target, what + ".json")
            ) as f:
                # append the results
                sets[what] = json.load(f)

        # set it on the target
        intermediate[target] = sets

    results.append({"settings": settings, "results": intermediate})
# [ { results: { target: { what: <base>}}} ]

# \begin{tabular}{|l|c|c|c|}\hline
#     \backslashbox{Label}{Bar} & \makebox[3em]{Time} & \makebox[3em]{Volume} & \makebox[3em]{Dollar}\\\hline
#     Movement & 0.046 & 0.041 & 0.040\\\hline
#     Up Movement & 0.023 & 0.021 & 0.021\\\hline
#     Anomaly & 0.018 & 0.003 & 0.003\\\hline
#     \end{tabular}

# order of things
upstream = [
    "InfoNCE",
    "VAE",
    "BCE-Movement",
    "BCE-Up-Movement",
    "BCE-Anomaly",
    "BCE-Future-Anomaly",
]


def find_experiment(loss, target, what, bartype="time"):
    for result in results:
        if result["settings"].loss == loss and result["settings"].bar_type == bartype:
            return result["results"][target][what]
    return None


for target in targets:
    # intiialize the graph
    rdat = np.zeros((6, 12))

    pos = {}

    # iterate over the upstream
    for idx, u in enumerate(upstream):
        for idxs, what in enumerate(splits):
            exp = find_experiment(u, target, what)
            rdat[idx][0 + idxs] = exp["accuracy"]
            rdat[idx][3 + idxs] = exp["precision"]
            rdat[idx][6 + idxs] = exp["recall"]
            rdat[idx][9 + idxs] = exp["phi"]
            pos[what] = exp["positive"]

    bold = np.argmax(rdat, axis=0)

    print("\n\n")
    print("\\begin{table}")
    print("\\centering")
    print("%% target is %s" % target)
    print("\\resizebox{\\columnwidth}{!}{%%")
    print("\\begin{tabular}{|l|ccc|ccc|ccc|ccc|}")
    print("\\hline")
    print(
        "\\backslashbox{upstream}{metric} & \\multicolumn{3}{c|}{accuracy} & \\multicolumn{3}{c|}{precision} & \\multicolumn{3}{c|}{recall} & \\multicolumn{3}{c|}{Phi}\\\\\\hline"
    )
    print(
        " & train & valid & test & train & valid & test & train & valid & test & train & valid & test\\\\\\hline"
    )

    for row in range(rdat.shape[0]):
        print(
            ("%s & " % upstream[row])
            + " & ".join(
                [
                    ("%.3f" if bold[idx] != row else "\\textbf{%.3f}") % val
                    for idx, val in enumerate(rdat[row])
                ]
            )
            + "\\\\"
        )

    print("\\hline")
    print("\\end{tabular}%%\n}")
    print(
        "\\caption{Classification results for the %s downstream classifiers on the time bar representations. Shown in bold is the best performance. Proportion of positive samples for the training set is %.3f, for the validation set is %.3f, and for the testing set is %.3f.}"
        % (target.replace("_", "-"), pos["train"], pos["valid"], pos["test"])
    )
    print("\\label{tab:result-time-%s}" % target)
    print("\\end{table}")
