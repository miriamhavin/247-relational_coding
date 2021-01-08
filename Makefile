FILE := tfsenc_main
USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

# E_LIST := $(shell seq 1 2)
SID := 625
NPERM := 1
LAGS := {-5000..5000..500}
EMB := gpt2
EMB_DIM := 768
EMB_RED_DIM := 50
WS := 200
CNXT_LEN := 512

MWF := 1
WV := all
# SH := --shuffle
# PSH := --phase-shuffle

CMD := python
# CMD := sbatch submit1.sh

# move paths to makefile
# electrode list
# add shuffle flag
# plotting modularity
# make separate models with separate electrodes (all at once is possible)
PDIR := $(shell dirname `pwd`)
link-data:
	find data/ -xtype l -delete
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
	$(CMD) code/$(FILE).py \
		--sid $(SID) \
		--electrodes $(E_LIST) \
		--emb-type $(EMB) \
		--window-size $(WS) \
		--word-value $(WV) \
		--glove $(GLOVE) \
		--gpt2 $(GPT2) \
		--npermutations $(NPERM) \
		--lags $(LAGS) \
		--sig-elec-file $(SE) \
		--min-word-freq $(MWF) \
		$(SH) \
		$(PSH) \
		--output-prefix cp-$(DT)-$(USR)-$(WS)ms-$(WV)-$(EMB)-$(EMB_DIM)d; \

plot-encoding:
	python code/tfsenc_plots.py \
		--sid $(SID) \
		--input-directory $(DT)-$(USR)-$(WS)ms-$(WV)-$(EMB)-$(SID) \
		--embedding-type $(EMB);

pca-on-embedding:
	python code/tfsenc_pca.py \
			--sid $(SID) \
			--emb-type $(EMB) \
			--context-length $(CNXT_LEN) \
			--reduce-to $(EMB_RED_DIM); 
