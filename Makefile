# -----------------------------------------------------------------------------
# Set up
# -----------------------------------------------------------------------------

PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

# commands
CMD := echo
CMD := python
CMD := sbatch submit1.sh


run-encoding:
	mkdir -p logs
	$(CMD) scripts/tfsenc_main.py \
		--config-file configs/config.yml configs/625-config.yml configs/gpt2-config.yml


SIDS:= 625 676 7170 798
run-encoding-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfsenc_main.py \
			--config-file configs/config.yml configs/$$sid-config.yml configs/glove-config-r.yml; \
	done;


run-erp:
	mkdir -p logs
	$(CMD) scripts/tfserp_main.py \
		--config-file configs/erp-config.yml configs/625-config.yml


run-erp-sids:
	mkdir -p logs
	for sid in $(SIDS); do \
		$(CMD) scripts/tfserp_main.py \
			--config-file configs/erp-config.yml configs/$$sid-config.yml; \
	done;

# -------- grid submit (subjects × lags × min_occ) --------
run-grid:
	mkdir -p logs
	sbatch --array=0-47%20 submit_grid.sh

# -------- merge all per-run CSVs to one Parquet + Pickle --------
merge-grid:
	python - <<'PY'
	import glob, os
	import pandas as pd

	roots = glob.glob("results/tfs/*/*") + glob.glob("results/podcast/*/*")
	csvs  = glob.glob("results/**/all_spaces_summary.csv", recursive=True)
	if not csvs:
		print("No all_spaces_summary.csv files found.")
		raise SystemExit(0)

	dfs = []
	for path in csvs:
		try:
			df = pd.read_csv(path)
			# add run tag from parent dir to keep provenance
			tag = os.path.basename(os.path.dirname(path))
			df["run_tag"] = tag
			dfs.append(df)
		except Exception as e:
			print(f"Skip {path}: {e}")

	big = pd.concat(dfs, ignore_index=True)
	out_dir = "results/combined"
	os.makedirs(out_dir, exist_ok=True)
	big.to_parquet(f"{out_dir}/all_spaces_summary.parquet", index=False)
	big.to_pickle(f"{out_dir}/all_spaces_summary.pkl")
	print(f"Wrote {len(big)} rows to {out_dir}/all_spaces_summary.(parquet|pkl)")
	PY
