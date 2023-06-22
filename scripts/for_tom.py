import os
import pandas as pd
import numpy as np
import glob


def main():
    sids = [625, 676, 7170, 798]
    elec_summary = pd.DataFrame(
        columns=[
            "elec",
            "patient",
            "prod_prob_max",
            "prod_prob_0",
            "prod_improb_max",
            "prod_improb_0",
            "comp_prob_max",
            "comp_prob_0",
            "comp_improb_max",
            "comp_improb_0",
        ]
    )

    for sid in sids:
        fdir = f"results/tfs/20230227-gpt2-preds/kw-tfs-full-{sid}-glove50-lag10k-25-gpt2-xl-prob/kw-200ms-all-{sid}/*_comp.csv"
        files = sorted(glob.glob(fdir))

        for resultfn in files:
            elec = os.path.basename(resultfn).replace(".csv", "")[:-5]
            elec = elec[elec.find("_") + 1 :]

            df = pd.read_csv(resultfn, header=None)
            prob_comp_max = df.loc[0, :].max()
            prob_comp_lag0 = df.loc[0, 400]

            prodfile = resultfn.replace("comp", "prod")
            df = pd.read_csv(prodfile, header=None)
            prob_prod_max = df.loc[0, :].max()
            prob_prod_lag0 = df.loc[0, 400]

            improbcompfile = resultfn.replace("prob", "improb")
            df = pd.read_csv(improbcompfile, header=None)
            improb_comp_max = df.loc[0, :].max()
            improb_comp_lag0 = df.loc[0, 400]

            improbprodfile = improbcompfile.replace("comp", "prod")
            df = pd.read_csv(improbprodfile, header=None)
            improb_prod_max = df.loc[0, :].max()
            improb_prod_lag0 = df.loc[0, 400]

            elec_summary.loc[elec] = [
                elec,
                sid,
                prob_prod_max,
                prob_prod_lag0,
                improb_prod_max,
                improb_prod_lag0,
                prob_comp_max,
                prob_comp_lag0,
                improb_comp_max,
                improb_comp_lag0,
            ]

    elec_summary.to_csv("results_for_tom.csv")
    breakpoint()

    return


if __name__ == "__main__":
    main()
