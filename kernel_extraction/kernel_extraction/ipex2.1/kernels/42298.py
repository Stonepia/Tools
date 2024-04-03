

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/nk/cnke7jrm3usnyztd5iirazjm6a7n37ek23hofz77432ozakyknqi.py
# Source Nodes: [add_6, getattr_getattr_l__mod___stages___2_____4___act1, mul_60, mul_61, mul_62, mul_63], Original ATen: [aten.add, aten.mul, aten.silu]
# add_6 => add_41
# getattr_getattr_l__mod___stages___2_____4___act1 => mul_164, sigmoid_38
# mul_60 => mul_161
# mul_61 => mul_162
# mul_62 => mul_163
# mul_63 => mul_165
triton_poi_fused_add_mul_silu_16 = async_compile.triton('triton_poi_fused_add_mul_silu_16', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_silu_16(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63700992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x3), None)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tmp11 = 0.9284766908852592
    tmp12 = tmp10 * tmp11
    tl.store(out_ptr0 + (x3), tmp12, None)
''')
