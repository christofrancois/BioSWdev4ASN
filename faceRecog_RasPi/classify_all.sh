for datafolder in $(du --max-depth=1 ./rawdata/ | sort -n -r | tail -n +2 | cut -f2)
do
#./rawdata/*/
  echo $datafolder
  python ./rawdata/classifydata.py $datafolder -c ./classifier/LinearSvm.pkl -v
done
