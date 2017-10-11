PIFACE="/worktmp/piface"

for image in $PIFACE/rawdata/*/*
do
	python $PIFACE/rawdata/classifyandmove.py $image -v -t 0
done

find $PIFACE/rawdata -empty -type d -delete
