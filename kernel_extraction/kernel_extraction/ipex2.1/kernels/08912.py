

# Original file: ./jx_nest_base___60.0/jx_nest_base___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/sp/cspcwmmbrlmyumrvlwqhj6rebqefjxcacw2klrpbh27wtql5szqf.py
# Source Nodes: [pad], Original ATen: [aten.constant_pad_nd]
# pad => constant_pad_nd
triton_poi_fused_constant_pad_nd_13 = async_compile.triton('triton_poi_fused_constant_pad_nd_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26615808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 14592) % 57
    x1 = (xindex // 256) % 57
    x3 = (xindex // 831744)
    x4 = xindex % 14592
    x0 = xindex % 256
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 56, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + (14336*x2) + (802816*x3)), tmp5, other=0.0).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (x1 + (56*x2) + (3136*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.load(in_ptr2 + (x1 + (56*x2) + (3136*x3)), tmp5, eviction_policy='evict_last', other=0.0)
    tmp11 = 256.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = tmp9 * tmp15
    tmp17 = tl.load(in_ptr3 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x0), tmp5, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tl.where(tmp5, tmp21, float("-inf"))
    tl.store(out_ptr0 + (x5), tmp22, None)
''')
