########################################################################################################################

.PHONY: test
test:
	python3 ./test/test_jit.py
	python3 ./test/test_som_abstract.py
	python3 ./test/test_som_pca.py
	python3 ./test/test_som_online.py
	python3 ./test/test_som_batch.py

########################################################################################################################
