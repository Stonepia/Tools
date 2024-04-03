

# Original file: ./tf_efficientnet_b0___60.0/tf_efficientnet_b0___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/wc/cwciyrz26e23boec6rr4ge4jd33dq7o2biubgpvicpluj6zkzzm5.py
# Source Nodes: [getattr_getattr_l__self___blocks___1_____0___bn1_act, pad_1], Original ATen: [aten.constant_pad_nd, aten.silu]
# getattr_getattr_l__self___blocks___1_____0___bn1_act => convert_element_type_28, mul_16, sigmoid_4
# pad_1 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_silu_8 = async_compile.triton('triton_poi_fused_constant_pad_nd_silu_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_silu_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_silu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156905472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10848) % 113
    x1 = (xindex // 96) % 113
    x3 = (xindex // 1225824)
    x4 = xindex % 10848
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 112, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (10752*x2) + (1204224*x3)), tmp5, other=0.0)
    tmp7 = tl.sigmoid(tmp6)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.where(tmp5, tmp9, 0.0)
    tl.store(out_ptr0 + (x5), tmp10, None)
''')