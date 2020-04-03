head -n 13383 uniprot-reviewed-yes-all-for303-nodup-22305.txt > neg-train-ID.txt
tail -n +13384 uniprot-reviewed-yes-all-for303-nodup-22305.txt > neg-vali-test-ID.txt
head -n 4461 neg-vali-test-ID.txt > neg-vali-ID.txt
tail -n +4462 neg-vali-test-ID.txt > neg-test-ID.txt

head -n 14915 prep-neg-18644.txt > neg-train.txt
tail -n +14916 prep-neg-18644.txt > neg-test.txt

(base) wless-user-172019209183:Refined_Dataset jintao$ awk '!seen[$0]++' 269-function.txt > nondup-269.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ awk '!seen[$0]++' 303.txt > nondup-303.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ head -n 22305 nondup-269.txt > nondup-269-final.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ head -n 17844 nondup-269-final.txt > 269-train.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ tail -n +17845 nondup-269-final.txt > 269-test.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ head -n 17844 nondup-303-final.txt > 303-train.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ tail -n +17845 nondup-303-final.txt > 303-test.txt
(base) wless-user-172019209183:Refined_Dataset jintao$ 

test the number of empty lines:
One way using grep:

grep -c "^$" file

