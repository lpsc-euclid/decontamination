#!/bin/sh

########################################################################################################################

rm -fr ./docs/
mkdir -p ./docs/

pdoc3 -c sort_identifiers=False -c latex_math=True --output-dir ./docs/ --force --html decontamination

sed -i '' 's/numpy\./np\./g' ./docs/decontamination/index.html
sed -i '' 's/typing\.//g' ./docs/decontamination/index.html

mv ./docs/decontamination/* ./docs/

rm -fr ./docs/decontamination/

########################################################################################################################
