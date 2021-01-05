mkdir -p data results

#echo 'ls $(pwd)/../247-pickling/results/*'

cd data
ln -s $(pwd)/../../247-pickling/results/* .

cd ..
