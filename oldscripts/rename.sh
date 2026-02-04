for i in {0..15}
do 
  for (( j=i+1; j<=15; j++ ))
  do 
    mv /scratch/groups/cslevin/eeganr/cylwater/cylwat_eval/nocorr/split/${i}_${j}_coin.lm /scratch/groups/cslevin/eeganr/cylwater/cylwat_eval/nocorr/split/${i}_${j}_coin.dat
  done 
  echo "Number: $i" 
done
