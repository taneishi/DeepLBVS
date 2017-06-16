.PHONY: chembl pcba

all: chembl pcba tox21

chembl: log/chembl/tf_models.log log/chembl/graph_conv.log
	@echo "ChEMBL dataset"
	@cat log/chembl/tf_models.log
	@cat log/chembl/graph_conv.log
	@echo

pcba: log/pcba/tf_models.log log/pcba/graph_conv.log
	@echo "PCBA dataset"
	@cat log/pcba/tf_models.log
	@cat log/pcba/graph_conv.log
	@echo

tox21: log/tox21/tf_models.log log/tox21/graph_conv.log
	@echo "Tox21 dataset"
	@cat log/tox21/tf_models.log
	@cat log/tox21/graph_conv.log
	@echo

log/chembl/tf_models.log: chembl/tf_models.py
	python chembl/tf_models.py

log/chembl/graph_conv.log: chembl/graph_conv.py
	python chembl/graph_conv.py
	
log/pcba/tf_models.log: pcba/tf_models.py
	python pcba/tf_models.py

log/pcba/graph_conv.log: pcba/graph_conv.py
	python pcba/graph_conv.py
	
log/tox21/tf_models.log: tox21/tf_models.py
	python tox21/tf_models.py

log/tox21/graph_conv.log: tox21/graph_conv.py
	python tox21/graph_conv.py
	
