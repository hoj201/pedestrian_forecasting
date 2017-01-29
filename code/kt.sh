for i in $(seq 0 $2)
do
  echo $i
  time python lin_time_test.py $1 $i >> f.txt
done
