This is a branch for collecting triton kernels from the benchmark suites.

# Usage
## Prepare your output_code.py
The output_code.py is generated when the torch debug flag is turned on, please add the following flag to your `pytorch/benchmarks/dynamo/common.py`.


```Python
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = True
```

In addition, please set the following environment flag if you need:

```
export DEBUG_DIR_VAR_NAME=/home/user/my_folder
```
By default, it will set `debug_dir_root = os.path.join(os.environ[DEBUG_DIR_VAR_NAME], "torch_compile_debug")`.


Then run the benchmarks as usual. This will take quite a long time. In case you just need the kernels, you could set this `-n10` to smaller number, 
this controls how many iterations the model run.

[inductor_xpu_test.sh](https://github.com/intel/intel-xpu-backend-for-triton/blob/ca69c4c0b4b09ec2f77c4ad410d06c55f2572b65/scripts/inductor_xpu_test.sh#L55)


## Extract kernels using extract_kernel.py
The `extract_kernel.py` contains three functions. Please see the docs in code for more detail.

- `extract_output_code(debug_dir, output_code_dir)` : This function will extract all `output_code.py` to output_code_dir and rename it with model name.
- `extract_all_kernels(output_code_dir, kernel_dir)` : This will extract all triton kernels out to `kernel_dir`.
- `randomly_projection_files(n, kernel_dir, projection_dir, long_kernel_portion=0.8, long_kernel_pool_portion=0.3)`.
   
    Randomly selects and copies a specified number of kernel files from the kernel directory to the projection directory.
    Let's say there are 10,000 kernels in total, we would like to select 100 kernels from them.
    The long_kernel_portion=0.8, so 100*0.8=80 long kernels will be selected from 10,000 * 0.3 = 3,000 long kernels.
    The remaining 20 kernels will be randomly selected from the rest of the kernels.


# `kernel_extraction` Folder Structure

- `model_output_codes`: The original output_code.py, it contains all the runnable python files for a model. You could directly run them with `python model.py`, or refer the kernel input/output info with it.
- `kernels`: The extracted kernels extracted from the `model_output_codes` folder. There may have duplications.
- `projections`: Randomly picked kernels from the `kernels` folder.