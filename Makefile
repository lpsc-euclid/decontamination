########################################################################################################################

all: manual
	echo 'Ok.'

########################################################################################################################

.PHONY: manual
manual:
	cd manual && make html

########################################################################################################################

.PHONY: test
test:
	python3 -m pytest ./test/

########################################################################################################################

.PHONY: cov
cov:
	NUMBA_DISABLE_JIT=1 python3 -m pytest --cov=decontamination --cov-report xml:coverage.xml ./test/

########################################################################################################################

.PHONY: htmlcov
htmlcov:
	NUMBA_DISABLE_JIT=1 python3 -m pytest --cov=decontamination --cov-report html ./test/

########################################################################################################################

clean:
	rm -fr ./doc/* ./build/ ./*.egg-info/ ./manual/_build/ ./.coverage* ./coverage.xml

########################################################################################################################
