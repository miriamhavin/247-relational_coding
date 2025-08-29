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
	python scripts/merge_grid.py

