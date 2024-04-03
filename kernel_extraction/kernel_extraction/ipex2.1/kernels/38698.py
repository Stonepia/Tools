

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/uh/cuhdyhzxweimmkxm5ilzqzrltimg7yunv2wbkusb4w4nr7gsky6r.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_25 = async_compile.triton('triton_poi_fused_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_25', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = 0.03166208416223526
    tmp6 = 0.02478819154202938
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tl.full([1], 3, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tl.full([1], 4, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = 0.057761259377002716
    tmp13 = -0.025631887838244438
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = -0.049163609743118286
    tmp16 = tl.where(tmp9, tmp15, tmp14)
    tmp17 = tl.where(tmp2, tmp7, tmp16)
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''')
