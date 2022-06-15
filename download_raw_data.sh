mkdir data
cd data
##### Download MS MARCO (v1) Passage Ranking Dataset #####
# Using the pre-processed data from RocketQA
wget --no-check-certificate https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
tar -zxf marco.tar.gz
rm -rf marco.tar.gz
mv marco raw
mkdir msmarco_passage
mv raw msmarco_passage
cd msmarco_passage/raw
mv dev.query.txt queries.dev.small.tsv

# Join the para and title as the final corpus (different from the official collections.tsv, which does not contain the title field)
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv

# Using official train triples (ids) file for (initial) negatives
wget https://msmarco.blob.core.windows.net/msmarcoranking/qidpidtriples.train.full.2.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
gunzip qidpidtriples.train.full.2.tsv.gz
awk -v RS='\r\n' '$1==last {printf ",%s",$3; next} NR>1 {print "";} {last=$1; printf "%s\t%s",$1,$3;} END{print "";}' qidpidtriples.train.full.2.tsv > train.negatives.tsv

#### MS MARCO Document Ranking Dataset: https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020#document-ranking-dataset
#### ANTIQUE Corpus: https://ciir.cs.umass.edu/downloads/Antique/antique-collection.txt