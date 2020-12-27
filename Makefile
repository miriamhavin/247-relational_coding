FILE := tfsenc_main
USR := $(shell whoami | head -c 2)

# E_LIST := $(shell seq 1 10)
SID := 625
NPERM := 500
LAGS := {-2000..2000..25}
EMB := glove
WS := 200
LAGS := {-2000..2000..100}
DT := $(shell date +"%Y%m%d")
WS := 200
GPT2 := 0
GLOVE := 0
MWF := 1
WV := 'all'
SH := --shuffle
# PSH := --phase-shuffle

CMD := python
CMD := sbatch submit1.sh

# move paths to makefile
# electrode list
# add shuffle flag
# plotting modularity
# make separate models with separate electrodes (all at once is possible)

target1:
	for elec in $(E_LIST); do \
		$(CMD) $(FILE).py \
			--subject $(SID) \
			--lags $(LAGS) \
			--emb-file $(EMB) \
			--electrode $$elec \
			--npermutations 
			--output-folder $(SID)-$(USR)-test1; \
	done

run-perm-cluster:
	mkdir -p logs
	$(CMD) $(FILE).py \
		--sid $(SID) \
		--electrodes $(E_LIST) \
		--datum-emb-fn $(EMB) \
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
		--output-prefix $(DT)-$(USR)-$(WS)ms-$(WV)-$(PIL); \

