import sys

input_file = sys.argv[1]

with open(input_file) as fin:
    lines = fin.readlines()

def make_file_with_topk(topk):
    new_data = ""
    for i in range(0, len(lines), 10):
        newlines = lines[i:i+topk]
        new_data += "".join(newlines)
    output_file = input_file.replace("10", str(topk))
    with open(output_file, "w") as fout:
        fout.write(new_data)

for k in [1, 3, 5]:
    make_file_with_topk(k)
