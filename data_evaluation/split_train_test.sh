#!/bin/bash

echo="generate_data"
types="STRING"
exe="Train_Test_Split.py"
# "STRING" "Specific"
for types in "Non-Specific"
do
  python3 $exe --net=$types --num=500 --data=hESC
  python3 $exe --net=$types --num=1000 --data=hESC
  
  python3 $exe --net=$types --num=500 --data=hHEP
  python3 $exe --net=$types --num=1000 --data=hHEP
  
  python3 $exe --net=$types --num=500 --data=mDC
  python3 $exe --net=$types --num=1000 --data=mDC
  
  python3 $exe --net=$types --num=500 --data=mESC
  python3 $exe --net=$types --num=1000 --data=mESC
  
  python3 $exe --net=$types --num=500 --data=mHSC-E
  python3 $exe --net=$types --num=1000 --data=mHSC-E
  
  python3 $exe --net=$types --num=500 --data=mHSC-GM
  python3 $exe --net=$types --num=1000 --data=mHSC-GM
  
  python3 $exe --net=$types --num=500 --data=mHSC-L
  python3 $exe --net=$types --num=1000 --data=mHSC-L
  echo = "$types TYPE all"
done
echo = "data split complete!"