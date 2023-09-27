import os
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.colors as mcolors

from itertools import cycle
lines = ["-", "--", "-.", ":"]
linecycler = cycle(lines)


# Function to process a single file and return the extracted data and the first line
def process_file(file_name):
    # Initialize an empty list to store the extracted data
    data_list = []

    # Initialize a variable to store the first line
    first_line = None

    # Open the file for reading
    with open(file_name, 'r') as file:
        lines = file.readlines()

        # Extract the first line
        if lines:
            first_line = lines[0].strip()

        for line in lines:
            # Check if the line contains four numbers (two integers and two floats)
            parts = line.split()
            if len(parts) == 4:
                try:
                    # Convert the parts to integers and floats
                    int1 = int(parts[0])
                    int2 = int(parts[1])
                    float1 = float(parts[2])
                    float2 = float(parts[3])

                    # Add the data to the list as a tuple
                    data_list.append((int1, int2, float1, float2))
                except ValueError:
                    pass  # Ignore lines that don't match the expected format


    # Return the extracted data and the first line for this file
    return data_list, first_line

# Directory containing the files (adjust this as needed)
directory = './data'

# Create a dictionary to store data and the first line from each file
data_dict = {}

# List all files in the directory that start with 'out' and end with '.txt'
#file_names = [filename for filename in os.listdir(directory) if filename.startswith('out') and filename.endswith('.txt')]
file_names= []
with open("data/filenames.txt", 'r') as file:
    file_names = [line.strip() for line in file.readlines()]


m_classes = set()

# Process each file in the list and store the data and the first line in the dictionary
for file_name in file_names:
    m_classes.add(re.search(r'(n\d+_m\d+)', file_name).group(1))
    file_path = os.path.join(directory, file_name)
    data, first_line = process_file(file_path)
    data_dict[file_name] = {'data': data, 'first_line': first_line}
m_classes = list(m_classes)
print(m_classes)
#class_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(m_classes)))
#color_map = dict(zip(m_classes, class_colors))

class_colors = mcolors.ListedColormap(plt.cm.Dark2(np.arange(len(m_classes))))
color_map = {key: idx for idx, key in enumerate(m_classes)}

# Create two plots: one with the 1st column as the domain and the last column as values,    //or three :D
# and another with the 3rd column as the domain and the last column as values

# Plot 1: 1st column vs. last column (Iteration count vs. Values) on a logarithmic scale
plt.figure(figsize=(10, 6))
plt.xlabel('Iteration Count')
plt.ylabel('Values (log scale)')

for file_name, file_data in data_dict.items():
    data_list = file_data['data']
    iteration_count = [data[0] for data in data_list]
    values = [data[3] for data in data_list]
    first_line = file_data['first_line']
    #plt.semilogy(iteration_count, values, label=first_line)

    file_class=re.search(r'(n\d+_m\d+)', file_name).group(1)
    color = class_colors(color_map[file_class])
    plt.semilogy(iteration_count, values,next(linecycler),color=color, label=file_name)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Iteration Count vs. Values (log scale)')
plt.savefig('graph1_log_legend.pdf', bbox_inches='tight')
plt.close()

# Plot 2: 3rd column vs. last column (Time vs. Values) on a logarithmic scale
plt.figure(figsize=(10, 6))
plt.xlabel('Time')
plt.ylabel('Values (log scale)')

for file_name, file_data in data_dict.items():
    data_list = file_data['data']
    time = [data[2] for data in data_list]
    values = [data[3] for data in data_list]
    first_line = file_data['first_line']
    #plt.semilogy(time, values, label=first_line)
    file_class=re.search(r'(n\d+_m\d+)', file_name).group(1)
    color = class_colors(color_map[file_class])
    plt.semilogy(time, values,next(linecycler),color=color, label=file_name)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Time vs. Values (log scale)')
plt.savefig('graph2_log_legend.pdf', bbox_inches='tight')
plt.close()



# Plot 3: 2nd column vs. last column (Iteration count vs. cg iters) 
plt.figure(figsize=(10, 6))
plt.xlabel('Time iterations')
plt.ylabel('Cg iterations')

for file_name, file_data in data_dict.items():
    data_list = file_data['data']
    iter = [data[0] for data in data_list]
    cg_iter = [data[1] for data in data_list]
    first_line = file_data['first_line']
    
    file_class=re.search(r'(n\d+_m\d+)', file_name).group(1)
    color = class_colors(color_map[file_class])
    plt.plot(iter, cg_iter,next(linecycler),color=color, label=file_name)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title('Timesteps vs. cg iterations')
plt.savefig('graph3_legend.pdf', bbox_inches='tight')
plt.close()



print("Logarithmic scale graphs with legends saved as graph1_log_legend.pdf and graph2_log_legend.pdf  and graph3_legend.pdf")
