from support.dataset_fb15k237 import Dataset_FB15k237
import sys

outf = sys.argv[1]

db = Dataset_FB15k237()

with open(outf, 'wt') as fout:
    fout.write("Entity\tRelation\tValue\tIteration of Promotion\tProbability\tSource\tCandidate Source\n")
    fout.write("concept:everypromotedthing\tgeneralizations	everything\t155\t1.0\t[OntologyModifier-Iter:155-2010/10/02-13:39:50-tsv_to_om_category.pl-categories.xls]\t[]\n")
    facts = db.get_facts()
    for fact in facts:
        h = db.get_entity_text(fact[0])
        t = db.get_entity_text(fact[1])
        r = db.get_relation_text(fact[2])
        #fout.write(h + '\t' + r + '\t' + t + '\t160\t1\t[]\t[]\n')
        break
    fout.close()