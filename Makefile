# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := tfs
# {podcast | tfs}

# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)

# # 676 Electrode IDs
# SID := 676
# E_LIST := $(shell seq 1 125)

PKL_IDENTIFIER := full
# {full | trimmed}

# podcast electeode IDs
# SID := 661
# E_LIST :=  $(shell seq 1 115)
# SID := 662
# E_LIST :=  $(shell seq 1 100)
# SID := 717
# E_LIST :=  $(shell seq 1 255)
# SID := 723
# E_LIST :=  $(shell seq 1 165)
# SID := 741
# E_LIST :=  $(shell seq 1 130)
# SID := 742
# E_LIST :=  $(shell seq 1 175)
# SID := 743
# E_LIST :=  $(shell seq 1 125)
# SID := 763
# E_LIST :=  $(shell seq 1 80)
# SID := 798
# E_LIST :=  $(shell seq 1 195)

# number of permutations (goes with SH and PSH)
NPERM := 1

# Choose the lags to run for.
LAGS := {-2000..2000..25}

CONVERSATION_IDX := 0

# Choose which set of embeddings to use
EMB := blenderbot-small
# {glove50 | gpt2-xl | blenderbot-small}
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
# SH := --shuffle
# PSH := --phase-shuffle

# Choose whether to normalize the embeddings
# NM := l2
# {l1 | l2 | max}

# PCA_TO := 50

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := sbatch submit1.sh
# {echo | python | sbatch submit1.sh}

# datum
# DS := podcast-datum-glove-50d.csv
# DS := podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv

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
			--output-folder $(DT)-$(SID)-test; \
	done

# Run the encoding model for the given electrodes in one swoop
# Note that the code will add the subject, embedding type, and PCA details to
# the output folder name
run-encoding:
	mkdir -p logs
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER) \
			--datum-emb-fn $(DS) \
			--sid $(SID) \
			--conversation-id $(CONVERSATION_IDX) \
			--electrodes $(E_LIST) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			--pca-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--normalize $(NM)\
			--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB) \
			--output-prefix '';\

run-sig-encoding:
	mkdir -p logs
		$(CMD) code/$(FILE).py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER) \
			--datum-emb-fn $(DS) \
			--conversation-id $(CONVERSATION_IDX) \
			--sig-elec-file bobbi.csv \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--align-with $(ALIGN_WITH) \
			--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
			--window-size $(WS) \
			--word-value $(WV) \
			--npermutations $(NPERM) \
			--lags $(LAGS) \
			--min-word-freq $(MWF) \
			--pca-to $(PCA_TO) \
			$(SH) \
			$(PSH) \
			--normalize $(NM)\
			--output-parent-dir sig-elec-test-tfs \
			--output-prefix '';\

# Recommended naming convention for output_folder
#--output-prefix $(USR)-$(WS)ms-$(WV); \

# Run the encoding model for the given electrodes __one at a time__, ideally
# with slurm so it's all parallelized.
run-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) code/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--sid $(SID) \
				--electrodes $$elec \
				--conversation-id $(CONVERSATION_IDX) \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				--pca-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--normalize $(NM) \
				--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(EMB)-pca$(PCA_TO) \
				--output-prefix ''; \
				# --job-id $(EMB)-$$jobid; \
		# done; \
	done;


run-sig-encoding-slurm:
	mkdir -p logs
	for elec in $(E_LIST); do \
		# for jobid in $(shell seq 1 1); do \
			$(CMD) code/$(FILE).py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--sig-elec-file bobbi.csv \
				--emb-type $(EMB) \
				--context-length $(CNXT_LEN) \
				--align-with $(ALIGN_WITH) \
				--align-target-context-length $(ALIGN_TGT_CNXT_LEN) \
				--window-size $(WS) \
				--word-value $(WV) \
				--npermutations $(NPERM) \
				--lags $(LAGS) \
				--min-word-freq $(MWF) \
				--pca-to $(PCA_TO) \
				$(SH) \
				$(PSH) \
				--output-parent-dir podcast-gpt2-xl-transcription \
				--output-prefix ''; \
				# --job-id $(EMB)-$$jobid; \
		# done; \
	done;

pca-on-embedding:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--pca-to $(EMB_RED_DIM); 

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
plot-encoding1:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			--input-directory \
				20210503-2040-tfs-full-625-blenderbot-small-test \
			--labels \
				blenderbot-small \
			--output-file-name \
				20210503-2040-tfs-full-625-blenderbot-small-test;
