

# Original file: ./DALLE2_pytorch__23_inference_63.3/DALLE2_pytorch__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/kv/ckvjtc7k44wtsmc6sc5tvwj52snq5foteyt3zdsrpatezo4ehbzh.py
# Source Nodes: [cat_7], Original ATen: [aten.cat]
# cat_7 => cat_1
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 133120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 260
    x2 = (xindex // 8320) % 8
    x3 = (xindex // 66560)
    x4 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (512*x1) + (133120*x3)), None)
    tmp1 = 16.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0 + (64*x4)), tmp2, None)
''')
