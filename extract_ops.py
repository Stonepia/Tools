# This is a Python script that finds and prints the matches of a regex pattern
# in a folder of Python files. It also removes and sorts the duplicates from the
# result file.

# Usage: python script.py input_folder
# Example: python script.py src

# Author: Stonepia, Copilot
# Date: 2024-02
# License: MIT

import re
import glob
import os
import logging
import argparse

# define the regex pattern
# pattern = r"(tl(\.\w+)+|triton_helpers\.\w+|\.\w+)(?=\()"

pattern = r"(tl(\.\w+)+|triton_helpers\.\w+|\.\w+)(?=\()|(tt(\.\w+)+|arith(\.\w+)+|scf(\.\w+)+|math(\.\w+)+|triton_gpu(\.\w+)+)"


# create a logger object
logger = logging.getLogger("regex_finder")
# set the logging level
logger.setLevel(logging.INFO)
# create a file handler object
file_handler = logging.FileHandler("result.log")
# create a formatter object
formatter = logging.Formatter("%(message)s")
# set the formatter for the file handler
file_handler.setFormatter(formatter)
# add the file handler to the logger
logger.addHandler(file_handler)

# create an argument parser object
parser = argparse.ArgumentParser(
    description="Find regex matches in Python files")
# add an argument for the input folder
parser.add_argument("--input_folder",
                    help="the folder containing the Python files",
                    default=".Tools/cache")
# parse the command line arguments
args = parser.parse_args()
# get the input folder from the arguments
input_folder = args.input_folder

extensions = ["*.py", "*.ttir", "*.ttgir"]
# extensions = ["*.py"]
# open the result file for writing and reading
with open("result.log", "w+") as result_file:
    # loop through all the files that match *.py in input_folder
    for ext in extensions:
        for file_name in glob.glob(os.path.join(input_folder, "**", ext), recursive=True):
            # print(f"filename {file_name}")
            # open the file for reading
            with open(file_name, "r") as file:
                # read the file content as a string
                content = file.read()
                # find all the matches of the regex in the content
                matches = re.findall(pattern, content)
                # print(f"matches is {matches}")
                # remove the duplicates from the matches using set
                matches = set(matches)
                # sort the matches alphabetically using sorted()
                matches = sorted(matches)
                # loop through the matches
                for match in matches:
                    # log the match to the result file
                    if ext == '*.py':
                        logger.info(match[0])
                    else:
                        logger.info(match[2])
        # break
    # read the result file content as a list of lines
    lines = result_file.readlines()
    # remove the duplicates from the lines using set
    lines = set(lines)
    # sort the lines alphabetically using sorted()
    lines = sorted(lines)
# open the result file for writing
with open("result.log", "w") as result_file:
    # loop through the lines
    for line in lines:
        # log the line to the result file
        logger.info(line.strip())
