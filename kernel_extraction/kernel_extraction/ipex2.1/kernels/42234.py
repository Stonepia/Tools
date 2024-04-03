

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/6s/c6s4ivrcyny5sv7fwacfvrhuwrxfirlrlaj3qyoyydiqsyzc5kk2.py
# Source Nodes: [add_4, getattr_getattr_l__self___stages___2_____2___act1, mul_44, mul_45, mul_46, mul_47], Original ATen: [aten.add, aten.mul, aten.silu]
# add_4 => add_31
# getattr_getattr_l__self___stages___2_____2___act1 => convert_element_type_121, convert_element_type_122, mul_124, sigmoid_28
# mul_44 => mul_121
# mul_45 => mul_122
# mul_46 => mul_123
# mul_47 => mul_125
triton_poi_fused_add_mul_silu_20 = async_compile.triton('triton_poi_fused_add_mul_silu_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_silu_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 63700992
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 497664)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.9622504486493761
    tmp14 = tmp12 * tmp13
    tl.store(out_ptr0 + (x3), tmp14, None)
''')
