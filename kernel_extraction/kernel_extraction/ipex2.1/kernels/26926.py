

# Original file: ./tf_efficientnet_b0___60.0/tf_efficientnet_b0___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/hb/chbu3pnfcyhgyqef7vln2v52gd4beeqnenrf45aj3uuxtoj6x4pw.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_23 = async_compile.triton('triton_poi_fused_23', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_23', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_23(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 2, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = -9.489059448242188e-05
    tmp8 = 0.0029144287109375
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = -0.0016937255859375
    tmp11 = tl.where(tmp4, tmp10, tmp9)
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.full([1], 5, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = 0.004730224609375
    tmp17 = 0.00145721435546875
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = 0.009521484375
    tmp20 = tl.where(tmp13, tmp19, tmp18)
    tmp21 = tl.where(tmp2, tmp11, tmp20)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''')
