#!/bin/bash
prefix=/mnt/huge_26TB/data/
prefix2=result_dim_spmmv_com/
postfix=/saved_graph/
postfix2=_result_
dsets=(amazon as_skitter ca_astroPh cit_patent com_dblp email_euall k21 ogb_product roadNet_CA soc_livejournal1 sx_stackoverflow web_berkstand wiki_talk ca_hollywood09 orkut reddit) 

numv=(400727 1696415 18772 3774768 317080 265214 2097152 2449029 1965206 4847571 2601977 52579682 685230 2394385)
# twitter dataset is not included, it does not have a saved_graph

for k in 2 4 8 16 32
do
    for j in `seq 0 4`
    do
        #for i in `seq 0 15`
        for i in 4;
        #for i in `seq 0 1`
        do
            #echo "${prefix}${dsets[i]}${postfix}"
            #echo "python3 spmmv.py --gdir ${prefix}${dsets[i]}${postfix} --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt""
            python3 spmmv.py --gdir ${prefix}${dsets[i]}${postfix} --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt"
            #python3 gspmm-exp.py --gdir ${prefix}${dsets[i]}${postfix} --feat ${k} >> ${prefix2}${dsets[i]}${postfix2}${k}".txt"
            #python3 gspmm-exp.py --gdir ${prefix}${dsets[i]}${postfix} --feat 32 >> ${prefix2}${dsets[i]}${postfix2}
            #./csrmv ${prefix}${dsets[i]}${postfix} >> ${prefix2}${dsets[i]}${postfix2}
        done
    done
done
  
