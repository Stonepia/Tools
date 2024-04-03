

# Original file: ./nfnet_l0___60.0/nfnet_l0___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/ww/cwwgenxmmjhgv6punwoqnfv6434bzrb5gtbxyfexlb5eawzbiwcn.py
# Source Nodes: [add_6, add_7, getattr_getattr_l__mod___stages___2_____5___act1, mul_60, mul_61, mul_62, mul_68, mul_69, mul_70, mul_71], Original ATen: [aten.add, aten.mul, aten.silu]
# add_6 => add_41
# add_7 => add_46
# getattr_getattr_l__mod___stages___2_____5___act1 => mul_184, sigmoid_43
# mul_60 => mul_161
# mul_61 => mul_162
# mul_62 => mul_163
# mul_68 => mul_181
# mul_69 => mul_182
# mul_70 => mul_183
# mul_71 => mul_185
triton_poi_fused_add_mul_silu_17 = async_compile.triton('triton_poi_fused_add_mul_silu_17', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_silu_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_silu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp8 = tl.load(in_ptr3 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.2
    tmp6 = tmp4 * tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 * tmp3
    tmp11 = tmp10 * tmp5
    tmp13 = tmp11 + tmp12
    tmp14 = tmp6 + tmp13
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = 0.9128709291752768
    tmp18 = tmp16 * tmp17
    tl.store(in_out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr0 + (x3), tmp18, None)
''')
