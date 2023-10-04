########################################################################################################################

.PHONY: doc
doc:
	rm -fr ./doc/*.html

	pdoc3 -c sort_identifiers=False -c latex_math=True --output-dir ./doc/ --force --html decontamination

	mv ./doc/decontamination/* ./doc/
	rm -fr ./doc/decontamination/

	sed -i '' 's/matplotlib\.axes\._axes\.Axes/plt\.Axes/g' ./doc/index.html
	sed -i '' 's/matplotlib\.figure\.Figure/plt\.Figure/g' ./doc/index.html

	sed -i '' 's/numpy\./np\./g' ./doc/index.html
	sed -i '' 's/typing\.//g' ./doc/index.html

########################################################################################################################

.PHONY: test
test:
	python3 ./test/test_jit.py
	python3 ./test/test_som_abstract.py
	python3 ./test/test_som_pca.py
	python3 ./test/test_som_online.py
	python3 ./test/test_som_batch.py

########################################################################################################################
