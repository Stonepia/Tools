

# Original file: ./regnety_002___60.0/regnety_002___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/sn/csnibagrmycnlh6aeetbuue52lqfnfwmqz2aalrbl7cwhsdfs2u3.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp7 = 0.1431884765625
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.full([1], 3, tl.int64)
    tmp11 = tmp0 < tmp10
    tmp12 = tl.where(tmp11, tmp8, tmp8)
    tmp13 = tl.where(tmp4, tmp9, tmp12)
    tmp14 = tl.full([1], 6, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.full([1], 5, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = 0.235107421875
    tmp19 = 0.173583984375
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tl.full([1], 7, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = 0.1463623046875
    tmp24 = 0.1416015625
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tl.where(tmp15, tmp20, tmp25)
    tmp27 = tl.where(tmp2, tmp13, tmp26)
    tl.store(out_ptr0 + (x0), tmp27, xmask)
''')
