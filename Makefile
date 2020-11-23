FILE := 247_encoding
USR := $(shell whoami | head -c 2)

# E_LIST := $(shell seq 1 104)
E_LIST := 3 45 70 21 65 110 48 6 17 36 37
E_LIST := $(shell seq 1)
SID := 676
LAGS := {-10000..10000..25}
EMB := glove.6B.50d.txt
WS := 50
BS := 

CMD := python


target1:
	for elec in $(E_LIST); do \
		$(CMD) $(FILE).py \
			--subject $(SID) \
			--lags $(LAGS) \
			--emb-file $(EMB) \
			--electrode $$elec \
			--output-folder $(SID)-$(USR)-test1; \
	done
