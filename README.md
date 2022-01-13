# tumor-classifier-heroku
machine learning algorithm for guessing tumor type from list of mutated genes
Script which imports tab-delimited text file containing mutation data from MSK-IMPACT
study, obtained by download from cBioportal. Text file was pre-processed by adding a column
listing the tumor type. The genelist consists of 87 genes. In the original file, genes not mutant
in a sample were indicated "NaN', and any alteration that was detected was explicitly described e.g. G12D.
To simplify, the input file identifies non-mutant genes in each sample as '0' and a mutation of any kind as '1'.
No effort has been made to exclude passenger mutations.
Approach: 5 classifiers are trained on 10,945 specimens, each with 87 genes that are either wild-type or mutant.
60 tumor types are included. User enters the list of genes mutated in unknown sample. Script then attempts to classify
unknown profile against each of the five classifiers. The report indicates the top candidate cancer type predicted by
each of the five classifiers. Then follows a table showing the probability estimated by each classifer that the unknown
sample represents each tumor type. This table is rank ordered by the average probability computed by each of the five
classifiers.

This version adapted to run on Heroku platform. A working version is at https://tumor-classifier.herokuapp.com/
