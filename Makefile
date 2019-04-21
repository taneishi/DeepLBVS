.PHONY: tox21

all: tox21

clean:
	$(RM) log/tox21/tf_models.log log/tox21/graph_conv.log

tox21: log/tox21/tf_models.log log/tox21/graph_conv.log
	@echo "Tox21 dataset"
	@echo "Multi-task DNN"
	@cat log/tox21/tf_models.log
	@echo "Graph-Convolution"
	@cat log/tox21/graph_conv.log
	@echo

log/tox21/tf_models.log: tox21/tf_models.py
	python tox21/tf_models.py

log/tox21/graph_conv.log: tox21/graph_conv.py
	python tox21/graph_conv.py
