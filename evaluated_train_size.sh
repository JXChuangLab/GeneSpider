echo="begin"
exe="main.py"
for data in hESC hHEP mDC mESC mHSC-E mHSC-GM mHSC-L
do
  for size in 1 0.8 0.6 0.4 0.2
  do
    for types in "STRING" "Specific" "Non-Specific"
    do
      python3 $exe --net=$types --num=500 --data=$data --train_size=$size
      python3 $exe --net=$types --num=1000 --data=$data -train_size=$size
    done
  done
done
echo = "complete!"