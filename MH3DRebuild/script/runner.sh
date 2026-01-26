#!/bin/bash
array=(${LD_LIBRARY_PATH//:/ })
for var in ${array[@]}
do
lib_path=$var
done
source  $lib_path/conda_environment/bin/activate
python $1 $2 $3 $4 $5 $6 $7 $8 $9
