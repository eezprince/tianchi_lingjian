import numpy as np


# input
removed_column_path = r"data/removed_column.txt"
prepared_data_path = r"data/prepared_data.txt"
onehot_column_path = r"data/onehot_column.txt"

# configure
training_data = False

# output
normalized_data_path = r"data/normalized_data.txt"
# normalized data file style:
#   each data per line, x_norm = (x - x_mean) / x_std, x_onehot ~= [0,1,0...0] \
#     y_norm = (y - y_mean) / y_std
schema_path = r"data/schema.txt"
# schema file style:
#   line 0: one hot column, split by '\t'
#   line 1: total column, including removed column and id column
#   other lines (from line 2):
#     if one hot: mapping order, split by '\t', like: A\tB
#     if float: mean\tstd, like 0.0\t11.0
#     if removed: empty

# function
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

# generate schema
if training_data:
    with open(schema_path, "w") as f_out:
        with open(onehot_column_path, "r") as f_in:
            onehot_list = f_in.read().splitlines()
            f_out.write("\t".join(onehot_list))
            f_out.write("\n")

        with open(removed_column_path, "r") as f_in:
            removed_list = f_in.read().splitlines()

        with open(prepared_data_path, "r") as f_in:
            datas = [line.split("\t") for line in f_in.read().splitlines()]
            column_count = len(datas[0])
            print("total: {}".format(column_count))
            f_out.write("{}\n".format(column_count))
            column_index = 0
            while column_index < column_count:
                if column_index % 500 == 0:
                    print("processing column {} / {}.".format(column_index, column_count))
                if str(column_index) in removed_list:
                    f_out.write("\n")
                    column_index += 1
                    continue
                if str(column_index) in onehot_list:
                    s = set()
                    for row_index in xrange(len(datas)):
                        value = datas[row_index][column_index]
                        s.add(value)
                    f_out.write("\t".join(s))
                else:
                    l = list()
                    for row_index in xrange(len(datas)):
                        value = datas[row_index][column_index]
                        if value != "NaN":
                            l.append(float(value))
                    mean = np.mean(l)
                    std = np.std(l)
                    if not isclose(std, 0) and not mean > 1e10:
                        f_out.write("{}\t{}".format(np.mean(l), np.std(l)))
                column_index += 1
                f_out.write("\n")


# generate normalized data based on schema
with open(prepared_data_path, "r") as f_in:
    datas = f_in.read().splitlines()

with open(schema_path, "r") as f_in:
    schemas = f_in.read().splitlines()
    print("schema length: {}.".format(len(schemas)))

onehot_column = schemas[0].split("\t")
total_column = int(schemas[1])

with open(normalized_data_path, "w") as f_out:
    for row_index, data in enumerate(datas):
        # if row_index % 50 == 0:
        print("processing row {} / {}.".format(row_index, len(datas)))
        items = data.split("\t")
        column_index = 0
        read_removed_count = 0
        feature = list()
        while column_index < total_column:
            schema = schemas[column_index + 2]
            if not schema:
                column_index += 1
                continue
            value = items[column_index - read_removed_count]
            # print(value)
            if str(column_index) in onehot_column:
                encoding = schema.split("\t")
                print(encoding)
                code = encoding.index(value)
                one_hot = [0] * code + [1] + [0] * (len(encoding) - code - 1)
                feature.extend(one_hot)
            else:
                mean, std = [float(x) for x in schema.split("\t")]
                if value != "NaN":
                    feature.append((float(value) - mean) / std)
                else:
                    feature.append(mean)
            column_index += 1
        f_out.write("\t".join([str(x) for x in feature]))
        f_out.write("\n")

