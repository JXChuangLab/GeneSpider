#! /bin/bash
echo="begin"
exe="main.py"

for types in "STRING" "Specific" "Non-Specific"
do
  python3 $exe --net=$types --num=500 --data=hESC --net_code=True
  python3 $exe --net=$types --num=1000 --data=hESC --net_code=True
  
  python3 $exe --net=$types --num=500 --data=hHEP --net_code=True
  python3 $exe --net=$types --num=1000 --data=hHEP --net_code=True
  
  python3 $exe --net=$types --num=500 --data=mDC --net_code=True
  python3 $exe --net=$types --num=1000 --data=mDC --net_code=True
  
  python3 $exe --net=$types --num=500 --data=mESC --net_code=True
  python3 $exe --net=$types --num=1000 --data=mESC --net_code=True
  
  python3 $exe --net=$types --num=500 --data=mHSC-E --net_code=True
  python3 $exe --net=$types --num=1000 --data=mHSC-E --net_code=True
  
  python3 $exe --net=$types --num=500 --data=mHSC-GM --net_code=True
  python3 $exe --net=$types --num=1000 --data=mHSC-GM --net_code=True
  
  python3 $exe --net=$types --num=500 --data=mHSC-L --net_code=True
  python3 $exe --net=$types --num=1000 --data=mHSC-L --net_code=True
done
