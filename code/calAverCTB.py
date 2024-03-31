# Open the text file
fold_nums = [i for i in range(1,8)]
results_list = []
weighted = [0,0,0]
macro= [0,0,0]
for i in fold_nums:
    # Assuming 'results.txt' is the file containing the results
    file_path = f'./results/fold_{i}/results.txt'
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract precision (p), recall (r), and F1-score (f) from the lines
    for line in lines:
        if "weighted avg" in line:
            parts = line.split()
            p = float(parts[2])
            r = float(parts[3])
            f = float(parts[4])
            weighted[0]+=p
            weighted[1]+=r
            weighted[2]+=f
        if "macro avg" in line:
            parts = line.split()
            p = float(parts[2])
            r = float(parts[3])
            f = float(parts[4])
            macro[0]+=p
            macro[1]+=r
            macro[2]+=f
    # Print the extracted values
weighted = ["{:.4f}".format(i/len(fold_nums)) for i in weighted]
macro = ["{:.4f}".format(i/len(fold_nums)) for i in macro]

print(weighted)
print(macro)