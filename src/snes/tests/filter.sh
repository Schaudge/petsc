
n=7
for testcase in "numonly" "symnum"; do
o=summary_${testcase}.m
echo "N=[1367631,11089567,89314623];" > $o
echo "ptap${n}=[" >> $o
grep "Gal l00" *${testcase}_*${n} | awk ' NR % 2 == 1 {print $6}' >> $o # 1 -> take first occurence awk counts from 1
echo "];" >> $o
echo "opt${n}=[" >> $o
grep "Opt l00" *${testcase}_*${n} | awk '{print $6}' >> $o
echo "];" >> $o
echo "squ${n}=[" >> $o
grep "Squ l00" *${testcase}_*${n} | awk '{print $6}' >> $o
echo "];" >> $o
done
