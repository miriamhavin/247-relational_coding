FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")

PRJCT_ID := podcast
PRJCT_ID := tfs

# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)

# 676 Electrode IDs
SID := 676
# E_LIST := $(shell seq 1 125)


CONVERSATION_IDX := 0
NPERM := 1000
LAGS := {-2000..2000..25}
EMB := glove50
WS := 200
CNXT_LEN := 1024
ALIGN_WITH := gpt2-xl
ALIGN_TGT_CNXT_LEN := 1024

MWF := 1
WV := all

# SH := --shuffle
PSH := --phase-shuffle
# PCA := --pca-flag
PCA_TO := 50

CMD := python
CMD := sbatch submit1.sh
# CMD := echo

# move paths to makefile

# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/

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
