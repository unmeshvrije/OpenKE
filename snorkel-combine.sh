#!/bin/sh

if [ "$#" -lt 2 ];
then
    echo "usage: $0 [fb15k237/dbpedia50] [head/tail]"
    exit
fi

DB=$1

P=$2

if [ "$3" = "a" ];
then
    #python snorkel_combine.py --db $DB --pred $P --abstain
    python snorkel_without_sub.py --db $DB --pred $P --abstain
else
    #python snorkel_combine.py --db $DB --pred $P
    python snorkel_without_sub.py --db $DB --pred $P
fi
