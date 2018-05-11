i=1
while [[ $i -lt 16 ]]
do
    rm $i.txt
    let i=$i+1
done
