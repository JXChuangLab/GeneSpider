echo="begin"
exe="main.py"
 for cell_size in 0 0.2 0.4 0.6 0.8
 do
  for data in hESC hHEP mDC mESC mHSC-E mHSC-GM mHSC-L
   do
     python3 $exe --net=STRING --num=500 --data=$data --cell_size=$cell_size --GPU_id=3
     python3 $exe --net=STRING --num=1000 --data=$data --cell_size=$cell_size --GPU_id=3
  done
done
echo = "complete!"
