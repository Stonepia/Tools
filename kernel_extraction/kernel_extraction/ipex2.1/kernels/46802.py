

# Original file: ./DALLE2_pytorch__21_inference_61.1/DALLE2_pytorch__21_inference_61.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/te/cteuoizrwpr7xpm2f2wmh6fzjmp75nkylyrcs7ex2gjlyfxfshgw.py
# Source Nodes: [cos, mul_1, sin], Original ATen: [aten.cos, aten.mul, aten.sin]
# cos => cos
# mul_1 => mul_2
# sin => sin
triton_poi_fused_cos_mul_sin_1 = async_compile.triton('triton_poi_fused_cos_mul_sin_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cos_mul_sin_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cos_mul_sin_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256)
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = x0
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 1.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp8 = -0.036118981850886994
    tmp9 = tmp7 * tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tmp1 * tmp10
    tmp12 = tl.sin(tmp11)
    tmp13 = tl.cos(tmp11)
    tl.store(out_ptr0 + (x0 + (512*x1)), tmp12, xmask)
    tl.store(out_ptr1 + (x0 + (512*x1)), tmp13, xmask)
''')
