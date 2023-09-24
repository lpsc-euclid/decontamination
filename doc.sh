#!/bin/sh

########################################################################################################################

rm -fr ./docs/*.html

pdoc3 -c sort_identifiers=False -c latex_math=True --output-dir ./docs/ --force --html decontamination

mv ./docs/decontamination/* ./docs/
rm -fr ./docs/decontamination/

########################################################################################################################

sed -i '' 's/numpy\./np\./g' ./docs/index.html
sed -i '' 's/typing\.//g' ./docs/index.html

########################################################################################################################
