

# Original file: ./functorch_maml_omniglot___60.0/functorch_maml_omniglot___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/nq/cnqcx6htgrv3yce44i5cryqkgm556i4o7nz32iyd7zkytlu2hlnc.py
# Source Nodes: [mod_10, mod_11, mod_9], Original ATen: [aten._native_batch_norm_legit, aten.max_pool2d_with_indices, aten.relu]
# mod_10 => relu_2
# mod_11 => max_pool2d_with_indices_2
# mod_9 => add_4, add_5, convert_element_type_4, convert_element_type_5, mul_4, mul_5, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__native_batch_norm_legit_max_pool2d_with_indices_relu_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (576*x1)), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (576*x1)), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (192 + x0 + (576*x1)), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (256 + x0 + (576*x1)), xmask).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''')
