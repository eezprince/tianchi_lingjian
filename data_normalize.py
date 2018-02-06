import numpy as np
from scipy import stats


# input
prepared_data_path = r"data/prepared_data.txt"
onehot_column_path = r"data/onehot_column.txt"


# output
normalized_data_path = r"data/normalized_data.txt"

# configure
training_data = True
if not training_data:
    prepared_data_path += "_test"
    normalized_data_path += "_test"

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
def isclose(a, b, rel_tol=1e-06, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

valid_threshold = 1

# generate schema
if training_data:
    with open(schema_path, "w") as f_out:
        with open(onehot_column_path, "r") as f_in:
            onehot_list = f_in.read().splitlines()
            f_out.write("\t".join(onehot_list))
            f_out.write("\n")

        with open(prepared_data_path, "r") as f_in:
            datas = [line.split("\t") for line in f_in.read().splitlines()]
            Y = [float(row[-1]) for row in datas]
            column_count = len(datas[0])
            print("total: {}".format(column_count))
            f_out.write("{}\n".format(column_count))
            column_index = 0
            empty_count = 0
            # duplicate set
            column_set = set()
            while column_index < column_count:
                if column_index % 500 == 0:
                    print("processing column {} / {}.".format(column_index, column_count))

                # id
                if column_index == 0:
                    f_out.write("\n")
                    column_index += 1
                    empty_count += 1
                    continue

                # check duplicate
                l_for_dup = list()
                for row_index in xrange(len(datas)):
                    value = datas[row_index][column_index]
                    l_for_dup.append(value)
                s_for_dup = " ".join(l_for_dup)
                if s_for_dup in column_set:
                    f_out.write("\n")
                    column_index += 1
                    empty_count += 1
                    continue
                column_set.add(s_for_dup)

                if str(column_index) in onehot_list:
                    s = set()
                    for row_index in xrange(len(datas)):
                        value = datas[row_index][column_index]
                        s.add(value)
                    f_out.write("\t".join(s))
                else:
                    l = list()
                    zero_indexes = list()
                    for row_index in xrange(len(datas)):
                        value = datas[row_index][column_index]
                        if value != "NaN":
                            l.append(float(value))
                            if isclose(float(value), 0):
                                zero_indexes.append(len(l) - 1)
                    std = mean = 0
                    if len(l) > 0:
                        mean = np.mean(l)
                        std = np.std(l)
                        mode = stats.mode(l)
                    if isclose(std, 0) or l.count(mode) > len(l)/2 \
                            or len(l) < len(datas)/2:
                        empty_count += 1
                    else:
                        # pearsonsimilar
                        if abs(mean) > valid_threshold:  # recalculate mean and std
                            zero_indexes.reverse()
                            for index in zero_indexes:
                                l.pop(index)
                        std = np.std(l)
                        if not isclose(std, 0):
                            X = [row[column_index] for row in datas]
                            X = [mean if x.lower() == "nan" else float(x) for x in X]
                            corr_dis = abs(np.corrcoef(X, Y)[0,1])
                            if corr_dis > 0.1:
                                f_out.write("{}\t{}".format(np.mean(l), std))
                            else:
                                empty_count += 1
                column_index += 1
                f_out.write("\n")
            print("total: {}, removed: {}, remain: {}".format(column_count, empty_count, column_count - empty_count))


# generate normalized data based on schema
with open(prepared_data_path, "r") as f_in:
    datas = f_in.read().splitlines()

with open(schema_path, "r") as f_in:
    schemas = f_in.read().splitlines()
    print("schema length: {}.".format(len(schemas)))

onehot_column = schemas[0].split("\t")
total_column = int(schemas[1])
if not training_data:  # test data don't have y
    total_column -= 1
with open(normalized_data_path, "w") as f_out:
    for row_index, data in enumerate(datas):
        if row_index % 50 == 0:
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
            value = items[column_index]
            if str(column_index) in onehot_column:
                encoding = schema.split("\t")
                value = value.replace(".0", "")
                if value in encoding:
                    code = encoding.index(value)
                    one_hot = [0] * code + [1] + [0] * (len(encoding) - code - 1)
                else:
                    one_hot = [0] * len(encoding)
                feature.extend(one_hot)
            else:
                mean, std = [float(x) for x in schema.split("\t")]
                if value != "NaN" or (abs(mean) > valid_threshold and isclose(float(value), 0)):
                    feature.append((float(value) - mean) / std)
                else:
                    feature.append(mean)
            column_index += 1
        f_out.write("\t".join([str(x) for x in feature]))
        f_out.write("\n")

