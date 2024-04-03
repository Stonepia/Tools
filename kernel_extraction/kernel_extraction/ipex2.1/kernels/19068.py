

# Original file: ./functorch_maml_omniglot___60.0/functorch_maml_omniglot___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/se/csenpz3zhefoy67u46n2ew6qqhgc7zhj7e7jfcq75uomrhz2zxpw.py
# Source Nodes: [self_5, self_6, self_7], Original ATen: [aten._native_batch_norm_legit, aten.max_pool2d_with_indices, aten.relu]
# self_5 => add_2, add_3, convert_element_type_7, convert_element_type_8, mul_2, mul_3, rsqrt_1, sub_1, var_mean_1
# self_6 => relu_1
# self_7 => max_pool2d_with_indices_1
triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 5
    x2 = (xindex // 320) % 5
    x3 = (xindex // 1600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (1408*x2) + (7744*x3)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (1408*x2) + (7744*x3)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (704 + x0 + (128*x1) + (1408*x2) + (7744*x3)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (768 + x0 + (128*x1) + (1408*x2) + (7744*x3)), xmask).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x4), tmp6, xmask)
''')
