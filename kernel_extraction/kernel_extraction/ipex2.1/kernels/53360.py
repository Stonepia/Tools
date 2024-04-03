

# Original file: ./resnest101e___60.0/resnest101e___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/z4/cz4n5xuemwetjxj6cakcam46dicrsxobrvwl32t2wq5kpyhtvdei.py
# Source Nodes: [mul_30, sum_62], Original ATen: [aten.mul, aten.sum]
# mul_30 => mul_417
# sum_62 => sum_93
triton_poi_fused_mul_sum_17 = async_compile.triton('triton_poi_fused_mul_sum_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_mul_sum_17(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x3 = (xindex // 512)
    x2 = (xindex // 131072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (512 + x0 + (1024*x3)), None)
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp1 - tmp3
    tmp5 = tl.exp(tmp4)
    tmp6 = tmp2 - tmp3
    tmp7 = tl.exp(tmp6)
    tmp8 = tmp5 + tmp7
    tmp9 = tmp5 / tmp8
    tmp10 = tmp0 * tmp9
    tmp12 = tmp7 / tmp8
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (x4), tmp14, None)
''')
