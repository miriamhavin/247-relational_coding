PDIR := $(shell dirname `pwd`)

# commands
CMD := echo
CMD := python
CMD := sbatch submit1.sh

# -----------------------------------------------------------------------------
# Set up
# -----------------------------------------------------------------------------

link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

run-encoding:
	mkdir -p logs
	$(CMD) scripts/$(FILE).py \
		--config-file config.yml
