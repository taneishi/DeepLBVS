.PHONY: chembl pcba

all: chembl pcba

chembl: log/chembl/tf_models.log log/chembl/graph_conv.log
	@echo "ChEMBL dataset"
	@cat log/chembl/tf_models.log
	@cat log/chembl/graph_conv.log

pcba: log/pcba/tf_models.log log/pcba/graph_conv.log
	@echo "PCBA dataset"
	@cat log/pcba/tf_models.log
	@cat log/pcba/graph_conv.log

log/chembl/tf_models.log: chembl/tf_models.py
	python chembl/tf_models.py

log/chembl/graph_conv.log: chembl/graph_conv.py
	python chembl/graph_conv.py
	
log/pcba/tf_models.log: pcba/tf_models.py
	python pcba/tf_models.py

log/pcba/graph_conv.log: pcba/graph_conv.py
	python pcba/graph_conv.py
	
