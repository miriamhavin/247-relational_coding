# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")
DT := ${USR}

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := tfs
PRJCT_ID := podcast
# {podcast | tfs}

# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)
E_LIST := 63
E_LIST := 1 7 11 13 14 17 18 20 22 24 25 27 29 30 31 32 33 35 36 37 38 39 40 41 42 45 46 47 49 50 51 52 54 56 57 58 59 62 63 64 65
E_LIST := 1 7 11 13 14 17 18 20 22 24 25 27 29 30 31 32 33 35 36 37 38 39 40 41 42 45 46 47 49 50 51 52 54 56 57

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

SID := 777

# number of permutations (goes with SH and PSH)
NPERM := 1

# Choose the lags to run for.
LAGS := {-2000..2000..25}

CONVERSATION_IDX := 0

# Choose which set of embeddings to use
# {glove50 | gpt2-xl | blenderbot-small}
EMB := blenderbot
EMB := blenderbot-small
EMB := glove50
EMB := gpt2-xl
CNXT_LEN := 1024

# Choose the window size to average for each point
WS := 200

# Choose which set of embeddings to align with
ALIGN_WITH := blenderbot-small
ALIGN_WITH := gpt2-xl
ALIGN_TGT_CNXT_LEN := 1024

# Specify the minimum word frequency
MWF := 5

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
# SH := --shuffle
# PSH := --phase-shuffle

# Choose whether to normalize the embeddings
# NM := l2
# {l1 | l2 | max}

PCA_TO := 50

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := sbatch submit1.sh
CMD := python
# {echo | python | sbatch submit1.sh}

# datum
# DS := podcast-datum-glove-50d.csv
# DS := podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv

#TODO: move paths to makefile

SIG_FN := --sig-elec-file test.csv
SIG_FN := --sig-elec-file 129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv

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
			$(SIG_FN) \
			$(SH) \
			$(PSH) \
			--normalize $(NM)\
			--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB) \
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
				--output-parent-dir $(PRJCT_ID)-$(PKL_IDENTIFIER)-$(EMB)-pca$(PCA_TO) \
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
			# --electrodes $(E_LIST) 
			# d-tfs-full-625-glove50-55 \
				d-tfs-full-625-gpt2-xl-55 \
				d-tfs-full-625-blenderbot-small-55 \
				d-tfs-full-625-blenderbot-55 
plot-encoding:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			$(SIG_FN) \
			--input-directory \
				d-podcast-full-777-gpt2-xl-nosh-pcafirst \
				bobbi-full-777-gpt2-xl \
			--labels \
				py.gpt2 mat.gpt2 \
			--output-file-name \
				pod-gpt2-nosh-pca
	rsync -av --delete results/figures ~/tigress/247-encoding-results

plot-encoding1:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--project-id $(PRJCT_ID) \
			--sid 777 \
			--input-directory \
				podcast-e_bb-d_blenderbot-small-w_200/ \
				podcast-e_bb-d_glove50-w_200/ \
				podcast-e_bb-d_gpt2-xl-w_200/ \
			--labels \
				bbot-small glove gpt2-xl \
			--output-file-name \
				podcast-test
	rsync -av --delete results/figures ~/tigress/247-encoding-results
