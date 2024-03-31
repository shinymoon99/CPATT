fold_nums = [1,2,3,4,5,6,7,8,9,10]
results_list = []
for i in fold_nums:
    # Assuming 'results.txt' is the file containing the results
    file_path = f'./results/fold_{i}/results.txt'

    # Initialize dictionaries to hold the results

    detailed_results = {'P': [], 'R': [], 'F': []}

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting the lines that contain the detailed results for P, R, and F
    for line in lines:
        if line.startswith('P:'):
            detailed_results['P'] = [float(x.strip()) for x in line[3:].strip().strip('[]').split()]
        elif line.startswith('R:'):
            detailed_results['R'] = [float(x.strip()) for x in line[3:].strip().strip('[]').split()]
        elif line.startswith('F:'):
            detailed_results['F'] = [float(x.strip()) for x in line[3:].strip().strip('[]').split()]
    results_list.append(detailed_results)

# print(results_list)


# Initialize dictionaries to hold sums and counts
sums = {'P': {'micro': 0, 'macro': 0}, 'R': {'micro': 0, 'macro': 0}, 'F': {'micro': 0, 'macro': 0}}
counts = {'P': {'micro': 0, 'macro': 0}, 'R': {'micro': 0, 'macro': 0}, 'F': {'micro': 0, 'macro': 0}}

# Iterate over the data to calculate sums and counts
for entry in results_list:
    for key in sums.keys():
        sums[key]['micro'] += entry[key][0]
        sums[key]['macro'] += entry[key][1]
        counts[key]['micro'] += 1
        counts[key]['macro'] += 1

# Calculate averages
averages = {key: {'micro': sums[key]['micro'] / counts[key]['micro'], 'macro': sums[key]['macro'] / counts[key]['macro']} for key in sums.keys()}

print("Average Precision (P):", averages['P'])
print("Average Recall (R):", averages['R'])
print("Average F1-score (F):", averages['F'])
