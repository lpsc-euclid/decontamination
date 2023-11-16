########################################################################################################################

all: doc
	echo 'Ok.'

########################################################################################################################

.PHONY: doc
doc:
	cd doc && make html

########################################################################################################################

.PHONY: test
test:
	python3 -m pytest ./test/

########################################################################################################################

.PHONY: cov
cov:
	USE_NUMBA_CPU=0 python3 -m pytest --cov=decontamination --cov-report xml:coverage.xml ./test/

########################################################################################################################

.PHONY: htmlcov
htmlcov:
	USE_NUMBA_CPU=0 python3 -m pytest --cov=decontamination --cov-report html ./test/

########################################################################################################################

clean:
	rm -fr ./build/ ./*.egg-info/ ./doc/_build/ ./coverage.xml

########################################################################################################################
