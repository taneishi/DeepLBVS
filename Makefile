# package

dist:
	mkdir -p code
	for i in DBN rbm mlp logistic_sgd; do cp ~/deep/code/$$i.py code/; done;
	zip -r ../DBN.zip DBN.py utils.py code
