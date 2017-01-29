for i in $(seq 0 $2)
do
  echo $i
  time python time_test.py $1 $i >> f.txt
done
