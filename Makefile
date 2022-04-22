# Non-configurable paramters. Don't touch.
FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")
DT := ${USR}

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PRJCT_ID := tfs
# {podcast | tfs}

# 625 Electrode IDs
SID := 625
E_LIST := $(shell seq 1 105)
BC := 

# 676 Electrode IDs
SID := 676
E_LIST := $(shell seq 1 125)
BC := --bad-convos 38 39

# 717 Electrode IDs
# SID := 7170
# E_LIST := $(shell seq 1 256)
# BC :=

# Sig file will override whatever electrodes you choose
SIG_FN := 
# SIG_FN := --sig-elec-file test.csv
# SIG_FN := --sig-elec-file 129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv
# SIG_FN := --sig-elec-file colton625.csv colton625.csv
# SIG_FN := --sig-elec-file tfs-sig-file-625-top-0.3-prod.csv tfs-sig-file-625-sig-0.3-comp.csv
# SIG_FN := --sig-elec-file 625-mariano-prod-new-53.csv 625-mariano-comp-new-30.csv # for sig-test
# SIG_FN := --sig-elec-file 676-mariano-prod-new-109.csv 676-mariano-comp-new-104.csv # for sig-test
# SIG_FN := --sig-elec-file tfs-sig-file-676-sig-1.0-prod.csv # for plotting
# SIG_FN := --sig-elec-file 7170-38.csv

PKL_IDENTIFIER := full
# {full | trimmed}

# podcast electrode IDs
# SID := 777
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
#
# SIG_FN := --sig-elec-file test.csv
# SIG_FN := --sig-elec-file 129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv
# SIG_FN := --sig-elec-file 160-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH_newVer.csv
# SIG_FN :=

# number of permutations (goes with SH and PSH)
NPERM := 1000

# Choose the lags to run for.
LAGS := {400000..500000..100} # lag400500-100
LAGS := {-150000..150000..100} # lag60-1k
LAGS := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000 # lag60-10k
LAGS := -150000 -120000 -90000 90000 120000 150000 # lag150-30k
LAGS := -300000 -250000 -200000 200000 250000 300000 # lag300-50k
LAGS := {-10000..10000..25} # lag10-25


CONVERSATION_IDX := 0

# Choose which set of embeddings to use
# {glove50 | gpt2-xl | blenderbot-small}
EMB := blenderbot
EMB := glove50
EMB := blenderbot-small
EMB := gpt2-xl
CNXT_LEN := 1024

# Choose the window size to average for each point
# For ERP, choose the window size (after onset - before onset)
# WS := 120000 # erp window (-60 to 60s)
WS := 200

# Choose which set of embeddings to align with
ALIGN_WITH := gpt2-xl
ALIGN_WITH := blenderbot-small
ALIGN_WITH := glove50
ALIGN_WITH := gpt2-xl blenderbot-small

# Choose layer
# {1 for glove, 48 for gpt2, 8 for blenderbot encoder, 16 for blenderbot decoder}
LAYER_IDX := 48

# Choose whether to PCA
# PCA_TO := 50

# Specify the minimum word frequency
MWF := 0

# TODO: explain this parameter.
WV := all

# Choose whether to label or phase shuffle
# SH := --shuffle
# PSH := --phase-shuffle

# Choose whether to normalize the embeddings
# NM := l2
# {l1 | l2 | max}

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD := echo
CMD := python
CMD := sbatch submit1.sh
# {echo | python | sbatch submit1.sh}

# datum
# DS := podcast-datum-glove-50d.csv
# DS := podcast-datum-gpt2-xl-c_1024-previous-pca_50d.csv

# datum modifications
DM := gpt2-pred
DM := incorrect
DM := correct
DM := all
DM := first-1-inters
DM := test-lag-ctx-128
DM := lag300-50k

# model modification (best-lag model, prod-comp reverse model)
MM := prod-comp
MM := best-lag
MM := pc-flip-best-lag-150
MM := pc-flip-best-lag
MM := 

#TODO: move paths to makefile

# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/
	ln -s /projects/HASSON/247/data/podcast-data/*.csv data/
	# ln -fs /scratch/gpfs/${USER}/247-pickling/results/* data/

# -----------------------------------------------------------------------------
# Encoding
# -----------------------------------------------------------------------------

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
		--window-size $(WS) \
		--word-value $(WV) \
		--npermutations $(NPERM) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		--pca-to $(PCA_TO) \
		--layer-idx $(LAYER_IDX) \
		--datum-mod $(DM) \
		--model-mod $(MM) \
		$(BC) \
		$(SIG_FN) \
		$(SH) \
		$(PSH) \
		--normalize $(NM)\
		--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-$(EMB)-$(DM) \
		--output-prefix $(USR)-$(WS)ms-$(WV);\

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
				--output-parent-dir $(PRJCT_ID)-$(PKL_IDENTIFIER)-$(EMB)-pca$(PCA_TO); \
				# --output-prefix ''; \
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



# Run erp for the given electrodes in one swoop
# Choose if the datum is split based on GPT2 prediction

run-erp:
	mkdir -p logs
	$(CMD) code/tfserp_main.py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--datum-emb-fn $(DS) \
		--sid $(SID) \
		--conversation-id $(CONVERSATION_IDX) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--context-length $(CNXT_LEN) \
		--align-with $(ALIGN_WITH) \
		--window-size $(WS) \
		--word-value $(WV) \
		--lags $(LAGS) \
		--min-word-freq $(MWF) \
		--layer-idx $(LAYER_IDX) \
		--datum-mod $(DM) \
		--normalize $(NM)\
		$(SIG_FN) \
		--output-parent-dir $(DT)-$(PRJCT_ID)-$(PKL_IDENTIFIER)-$(SID)-erp-$(DM) \
		--output-prefix $(USR)-$(WS)ms-$(WV);\

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

plot-encoding:
	mkdir -p results/figures
	python code/tfsenc_plots.py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			--electrodes $(E_LIST) \
			--lags $(LAGS) \
			$(SIG_FN) \
			--input-directory \
				zz-tfs-full-625-glove50 \
			--labels \
				glove \
			--output-file-name \
				$(PRJCT_ID)-$(SID)-glove-one
	rsync -av results/figures/ ~/tigress/247-encoding-results/figures/

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

# 'results/tfs/zz1-tfs-full-625-blenderbot-small/625/*_%s.csv' 

# plot order: glove (blue), gpt2 (orange), decoder (green), encoder (red)

plot-new:
	rm -f results/figures/*
	python code/plot.py \
		--formats \
			'results/tfs/kw-tfs-full-676-glove50-lag10-25-interss/kw-200ms-all-676/*_%s.csv' \
		--labels glove \
		--values $(LAGS) \
		--keys prod comp \
		$(SIG_FN) \
		--outfile results/figures/tfs-676-test.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-old:
	rm -f results/figures/*
	python code/plot_old.py \
		--formats \
			'results/tfs/kw-tfs-full-7170-glove50-final/kw-200ms-all-7170/*_%s.csv' \
			'results/tfs/kw-tfs-full-7170-blenderbot-small-final/kw-200ms-all-7170/*_%s.csv' \
		--labels glove bbot \
		--values $(LAGS) \
		--keys prod \
		$(SIG_FN) \
		--outfile results/figures/tfs-7170-gb-final-prod.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/

plot-all:
	rm -f results/figures/*
	python code/plot_all.py \
		--formats \
			'results/tfs/kw-tfs-full-625-glove50-final/kw-200ms-all-625/*_%s.csv' \
			'results/tfs/kw-tfs-full-625-gpt2-xl-final/kw-200ms-all-625/*_%s.csv' \
			'results/tfs/kw-tfs-full-625-blenderbot-small-final/kw-200ms-all-625/*_%s.csv' \
			'results/tfs/kw-tfs-full-625-gpt2-xl-ctx-128-final/kw-200ms-all-625/*_%s.csv' \
			'results/tfs/kw-tfs-full-676-glove50-final/kw-200ms-all-676/*_%s.csv' \
			'results/tfs/kw-tfs-full-676-gpt2-xl-final/kw-200ms-all-676/*_%s.csv' \
			'results/tfs/kw-tfs-full-676-blenderbot-small-final/kw-200ms-all-676/*_%s.csv' \
			'results/tfs/kw-tfs-full-676-gpt2-xl-ctx-128-final/kw-200ms-all-676/*_%s.csv' \
		--labels glove gpt2-xl-1024 bbot-de gpt2-xl-128 \
		--values $(LAGS) \
		--keys prod \
		$(SIG_FN) \
		--sid 625 676 \
		--outfile results/figures/tfs-gggb-final-sig1.0-prod.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


plot-erp:
	rm -f results/figures/*
	python code/plot_erp.py \
		--formats \
			'results/tfs/kw-tfs-full-625-erp-quardra/kw-200ms-all-625/*_%s.csv' \
			'results/tfs/kw-tfs-full-625-gpt2-xl-det-quardra/kw-200ms-all-625/*_%s.csv' \
			'results/tfs/kw-tfs-full-625-blenderbot-small-det-quardra/kw-200ms-all-625/*_%s.csv' \
		--labels erp gpt2 bbot \
		--values $(LAGS) \
		--keys prod comp \
		$(SIG_FN) \
		--outfile results/figures/tfs-625-erp-quardra.pdf
	rsync -av results/figures/ ~/tigress/247-encoding-results/


SP := 1
# LAGS := -150000 -120000 -90000 -60000 -50000 -40000 -30000 -20000 {-10000..10000..25} 20000 30000 40000 50000 60000 90000 120000 150000

sig-test:
	rm -f results/figures/*
	python code/sig_test.py \
		--sid $(SID) \
		--formats \
			'results/tfs/kw-tfs-full-676-glove50-triple/kw-200ms-all-676/*_%s.csv' \
		--labels glove \
		--keys prod comp \
		--values $(LAGS) \
		$(SIG_FN) \
		--sig-percents $(SP)


# make sure the lags and the formats are in the same order
LAGS1 := {-10000..10000..25}
LAGS2 := -60000 -50000 -40000 -30000 -20000 20000 30000 40000 50000 60000
LAGS3 := -150000 -120000 -90000 90000 120000 150000
LAGS4 := -300000 -250000 -200000 200000 250000 300000
LAGS_FINAL := -99999999 # select all the lags that are concatenated
LAGS_FINAL := -300000 -30000 -60000 {-10000..10000..25} 30000 60000 300000

concat-lags:
	python code/concat_lags.py \
		--formats \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag10-25/kw-200ms-all-7170/' \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag60-10k/kw-200ms-all-7170/' \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag150-30k/kw-200ms-all-7170/' \
			'results/tfs/kw-tfs-full-7170-gpt2-xl-lag300-50k/kw-200ms-all-7170/' \
		--lags \
			$(LAGS1) \
			$(LAGS2) \
			$(LAGS3) \
			$(LAGS4) \
		--lags-final $(LAGS_FINAL) \
		--output-dir results/tfs/kw-tfs-full-7170-gpt2-xl-final/kw-200ms-all-7170/

plot-autocor:
	$(CMD) code/test.py

