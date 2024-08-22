import pandas as pd

tab = pd.read_excel("./result_summary.xlsx")
tab = tab[tab.version != "v2"]

repl = {
    "v1": "Manual", "v3": "muSAM (Generalist)", "v4": "CellPose (Default)",
    "v5": "muSAM (Default)", "v6": "CellPose (HIL)", "v7": "muSAM (Finetuned)", "v8": "CellPose (Finetuned)"
}
tab["version"] = tab["version"].replace(repl)
tab = tab.sort_values(by="version")

tab.to_excel("for_figure.xlsx", index=False)
