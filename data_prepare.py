# input

# output
prepared_data_path = r"data/prepared_data.txt"
onehot_column_path = r"data/onehot_column.txt"

# configure
training_data = False
if not training_data:
    prepared_data_path += "_test"
    data_path = r"data/testA.txt"
else:
    data_path = r"data/full_data.txt"


def read_file_as_list(file_path):
    with open(file_path, "r") as f_in:
        return f_in.read().splitlines()
# define some variable and read data to lines
datas = []
tool_column = []
with open(data_path, "r") as f_in:
    lines = f_in.read().splitlines()


# find tool column
if training_data:
    for index, title in enumerate(lines[0].split("\t")):
        if "tool" in title.lower():
            tool_column.append(index)
    # write one-hot colum
    with open(onehot_column_path, "w") as f_out:
        f_out.write("\n".join([str(i) for i in tool_column]))
else:
    tool_column = read_file_as_list(onehot_column_path)
print(" ".join(str(tool_column)))


# prepare datas
for row, line in enumerate(lines[1:]):
    data = []
    items = line.split("\t")
    for column, item in enumerate(items):
        item = item.strip()
        if not item or item.startswith("2017") or item.startswith("2.017"):
            data.append("NaN")
        elif str(column) in tool_column:
            data.append(item)
        else:
            value = item
            try:
                value = float(item)
            except ValueError:
                if column != 0:
                    print("'{}' at {}, {}.".format(item, row + 1, column))
            data.append(value)
    datas.append(data)


# write prepared data
with open(prepared_data_path, "w") as f_out:
    for data in datas:
        f_out.write("\t".join([str(d) for d in data]))
        f_out.write("\n")

