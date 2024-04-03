import os
import re
import random
import shutil


def extract_output_code(debug_dir, output_dir):
    """
    Extracts and renames the output_code.py files from the debug directory.

    Args:
        debug_dir (str): Path to the debug directory.
        output_dir (str): Path to the output directory.

    Returns:
        None
    """
    # Step 1: Enter the debug directory
    os.chdir(debug_dir)

    # Step 2: Extract and rename the output_code.py files
    count = 0

    for root, dirs, files in os.walk('.', topdown=True):
        for dir_name in dirs:
            model_name = dir_name.split('/')[-1]
            subfolder = os.path.join(output_dir, model_name)
            os.makedirs(subfolder, exist_ok=True)
            source_file = os.path.join(root, dir_name, 'output_code.py')
            if os.path.isfile(source_file):
                count += 1
                dest_file = os.path.join(subfolder, f'{model_name}.py')
                suffix = 0
                while os.path.isfile(dest_file):
                    dest_file = os.path.join(subfolder,
                                             f'{model_name}_{suffix}.py')
                    suffix += 1
                os.makedirs(
                    subfolder,
                    exist_ok=True)  # Create the subfolder if it doesn't exist
                shutil.copyfile(source_file, dest_file)

    # Step 3: Output the total number of output_code.py files extracted
    print(f'Total number of output_code.py files extracted: {count}')


def extract_all_kernels(output_code_dir, kernel_dir):
    """
    Extracts kernels from python files in the output_code directory and saves them in separate files.

    Args:
        output_code_dir (str): Path to the output_code directory.
        kernel_dir (str): Path to the kernel directory.

    Returns:
        int: Total number of kernels extracted.
    """
    # Step 1: Enter the folder 'output_code_dir' and get the list of python files
    folder_path = output_code_dir
    output_folder = kernel_dir
    os.chdir(folder_path)
    python_files = [
        os.path.join(root, f) for root, dirs, files in os.walk('.')
        for f in files if f.endswith('.py')
    ]
    # Step 2: Extract kernels from each python file and save them in separate files
    kernel_pattern = r"# kernel path: [\s\S]+?# Source Nodes: [\s\S]+?triton_[\w\d]+ = async_compile\.triton\('[\s\S]+?'''[\s\S]+?'''\)"
    os.makedirs(output_folder, exist_ok=True)
    print(f'Total number of python files: {len(python_files)}')
    total_count = 0
    for file_name in python_files:
        with open(file_name, 'r') as file:
            content = file.read()
            kernels = re.findall(kernel_pattern, content, re.MULTILINE)
            for kernel in kernels:
                output_file = os.path.join(
                    output_folder, f"{str(total_count+1).zfill(5)}.py")
                total_count = total_count + 1
                with open(output_file, 'w') as output:
                    output.write(f"\n\n# Original file: {file_name}\n\n")
                    output.writelines("import torch\n")
                    output.writelines("import intel_extension_for_pytorch\n")
                    output.writelines(
                        "from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile\n"
                    )
                    output.writelines("async_compile = XPUAsyncCompile()\n")
                    output.write('\n')
                    output.write(kernel)
                    output.write('\n')
    print(f'Total number of kernels extracted: {total_count}')
    return total_count


import matplotlib.pyplot as plt
import numpy as np

def randomly_projection_files(n, kernel_dir, projection_dir, remove_old_dir=True):
    """
    Randomly selects and copies a specified number of kernel files from the kernel directory to the projection directory based on the log10 hist.

    Args:
        n (int): Number of files to randomly select.
        kernel_dir (str): Path to the kernel directory.
        projection_dir (str): Path to the projection directory.
        remove_old_dir (bool, optional): Flag to remove the existing projection directory. Default is True.

    Returns:
        None
    """
    if remove_old_dir and os.path.exists(projection_dir):
        shutil.rmtree(projection_dir)
    os.makedirs(projection_dir)
    kernel_files = os.listdir(kernel_dir)
    total_files = len(kernel_files)
    file_sizes = []
    for file_name in kernel_files:
        file_path = os.path.join(kernel_dir, file_name)
        file_size = os.path.getsize(file_path)
        file_sizes.append(file_size)
    plt.hist(np.log10(file_sizes), bins=30)
    plt.xlabel('Logarithm of File Size')
    plt.ylabel('Frequency')
    plt.title('Histogram of Logarithm of File Sizes')
    hist, bin_edges = np.histogram(np.log10(file_sizes), bins=30)
    bin_weights = hist / np.sum(hist)
    selected_files = []
    for i in range(len(bin_edges) - 1):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        files_in_range = [file for file in kernel_files if lower_bound <= np.log10(os.path.getsize(os.path.join(kernel_dir, file))) < upper_bound]
        num_files_in_range = int(n * bin_weights[i])
        if len(files_in_range) > 0:
            selected_files += random.sample(files_in_range, max(num_files_in_range, 1))
            print(f"Log10 range [{lower_bound:.1f}, {upper_bound:.1f}] files_in_range: {len(files_in_range)}, num_files selected: {max(num_files_in_range, 1)}")

    for file_name in selected_files:
        source_file = os.path.join(kernel_dir, file_name)
        destination_file = os.path.join(projection_dir, file_name)
        shutil.copyfile(source_file, destination_file)
    plt.text(0.5, 0.5, f'{len(selected_files)} files randomly picked from {total_files} files.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.savefig('histogram.png')
    print(f'{len(selected_files)} files randomly picked from {total_files} files.')
    print(f'Output projected folder to {projection_dir}')


# NOTE : Please change these things according to your directory structure
# debug_dir = os.path.join(os.getcwd(), '/home/sdp/tongsu/pytorch/torch_compile_debug')
debug_dir = os.path.join('/home/sdp/tongsu/pytorch/torch_compile_debug')
output_code_dir = os.path.join(os.getcwd(), './kernel_extraction/ipex2.1/model_output_codes')
kernel_dir = os.path.join(os.getcwd(), './kernel_extraction/ipex2.1/kernels')
projection_dir = os.path.join(os.getcwd(), './kernel_extraction/ipex2.1/projections')

# extract_output_code(debug_dir, output_code_dir)
# total_kernel_numbers = extract_all_kernels(output_code_dir, kernel_dir)
randomly_projection_files(100, kernel_dir, projection_dir)
