#!/bin/sh

########################################################################################################################

rm -fr ./docs/
mkdir -p ./docs/

pdoc3 -c sort_identifiers=False -c latex_math=True --output-dir ./docs/ --force --html decontamination

mv ./docs/decontamination/* ./docs/

rm -fr ./docs/decontamination/

########################################################################################################################
