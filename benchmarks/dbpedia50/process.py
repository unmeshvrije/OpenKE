def build_dict(filename):
    table = {}
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            string = line.split()[0]
            number = line.split()[1]
            table[string] = number
    return table

ent_dict = build_dict("entity2id.txt")
rel_dict = build_dict("relation2id.txt")

def fun(filename):
    outfile = filename.split('.')[0]+"2id.txt"
    data = ""
    with open(filename, "r") as fin:
        lines = fin.readlines()
        data += str(len(lines)) + "\n"
        for line in lines:
            head_str = line.split()[0]
            tail_str = line.split()[1]
            rel_str  = line.split()[2]
            data += ent_dict[head_str] + " " + ent_dict[tail_str] + " " + rel_dict[rel_str] + "\n"
    with open(outfile, "w") as fout:
        fout.write(data)
# process all files

fun("train.txt")
fun("valid.txt")
fun("test.txt")

