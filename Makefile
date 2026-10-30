.PHONY: install data local-exp1 local-exp2 local-exp3 modal-exp1 modal-exp2 modal-exp3 compare clean
PYTHON ?= python3
install:
	$(PYTHON) -m pip install -r requirements.txt
data:
	$(PYTHON) -m src.prepare_data --all
local-exp1:
	$(PYTHON) train.py --config configs/exp1_baseline.yaml
local-exp2:
	$(PYTHON) train.py --config configs/exp2_preproc1.yaml
local-exp3:
	$(PYTHON) train.py --config configs/exp3_preproc2.yaml
modal-exp1:
	modal run train_modal.py --config configs/exp1_baseline.yaml
modal-exp2:
	modal run train_modal.py --config configs/exp2_preproc1.yaml
modal-exp3:
	modal run train_modal.py --config configs/exp3_preproc2.yaml
compare:
	$(PYTHON) -m src.evaluate --compare
clean:
	rm -rf runs/ __pycache__/ src/__pycache__/ *.pt