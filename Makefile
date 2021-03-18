# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := podcast
PRJCT_ID := tfs

# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)

# 676 Electrode IDs
SID := 676
E_LIST := $(shell seq 1 125)

# number of permutations (goes with SH and PSH)
NPERM := 1000

# Choose the lags to run for.
LAGS := {-5000..5000..25}

CONVERSATION_IDX := 0

# Choose which set of embeddings to use
EMB := glove50
EMB := gpt2-xl
CNXT_LEN := 1024

# Choose the window size to average for each point
WS := 200

# Choose which set of embeddings to align with
ALIGN_WITH := gpt2-xl
ALIGN_TGT_CNXT_LEN := 1024

# Specify the minimum word frequency
MWF := 1

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
SH := --shuffle
PSH := --phase-shuffle


# Choose whether to PCA the embeddings before regressing or not
PCA := --pca-flag
PCA_TO := 50

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := python
CMD := sbatch submit1.sh
CMD := echo

#TODO: move paths to makefile

# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------
target1:
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE).py \
			--subject $(SID) \
			--lags $(LAGS) \
			--emb-file $(EMB) \
			--electrode $$elec \
			--npermutations 
			--output-folder $(SID)-$(USR)-test1; \
	done

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-encoding:
	mkdir -p logs
	for elec in $(E_LIST); do \
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			--conversation-id $(CONVERSATION_IDX) \
			--electrodes $$elec \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			$(PCA) \
			--reduce-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--output-parent-dir colton-phase-shuffle \
			--output-prefix '';\
	done;

# Recommended naming convention for output_folder
#--output-prefix $(USR)-$(WS)ms-$(WV); \

# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		for jobid in $(shell seq 1 1); do \
			$(CMD) code/$(FILE).py \
				--sid $(SID) \
				--electrodes $$elec \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				$(PCA) \
				--reduce-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--output-prefix test-$(DT)-$(USR)-$(WS)ms-$(WV) \
				--job-id $$jobid; \
		done; \
	done;

pca-on-embedding:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--reduce-to $(EMB_RED_DIM); 

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plot-encoding1:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--sid $(SID) \
			--input-directory \
				20210114-0845-hg-200ms-all-676-gpt2-cnxt-1024-pca_0d \
				20210114-0845-hg-200ms-all-676-gpt2-cnxt-1024-pca_50d \
				20210114-0844-hg-200ms-all-676-glove50-cnxt-0-pca_0d \
			--labels \
				gpt2-Fcnxt-pca0d \
				gpt2-Fcnxt-pca50d \
				glove50 \
			--output-file-name \
				'$(DT)-$(SID)-glove50_gpt2-xl_Fcnxt-gpt2-xl_Fcnxt-pca_50d';
