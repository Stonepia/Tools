

# Original file: ./ghostnet_100___60.0/ghostnet_100___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/q6/cq64uyrtlst7bqrvpctdbiyxegzxl6vtroxdzll4pbk4uwxn22ur.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = -1.718287467956543
    tmp8 = 2.183924436569214
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = 5.70352840423584
    tmp13 = 3.7477757930755615
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.where(tmp4, tmp9, tmp14)
    tmp16 = tl.full([1], 6, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.full([1], 5, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = 0.35571742057800293
    tmp21 = 6.428923606872559
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tl.full([1], 7, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = -11.534069061279297
    tmp26 = 4.4968414306640625
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp17, tmp22, tmp27)
    tmp29 = tl.where(tmp2, tmp15, tmp28)
    tl.store(out_ptr0 + (x0), tmp29, xmask)
''')
