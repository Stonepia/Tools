

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/4p/c4phdkbpkfneeqdvgagnpvu6hwck5eulgudrt3rpfshzitmpwals.py
# Source Nodes: [add_8, getattr_getattr_l__mod___stages___3_____0___act1, mul_76, mul_77, mul_78, mul_79], Original ATen: [aten.add, aten.mul, aten.silu]
# add_8 => add_51
# getattr_getattr_l__mod___stages___3_____0___act1 => mul_204, sigmoid_48
# mul_76 => mul_201
# mul_77 => mul_202
# mul_78 => mul_203
# mul_79 => mul_205
triton_poi_fused_add_mul_silu_18 = async_compile.triton('triton_poi_fused_add_mul_silu_18', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_18', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_silu_18(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63700992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp8 * tmp9
    tmp11 = 0.8980265101338745
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')
