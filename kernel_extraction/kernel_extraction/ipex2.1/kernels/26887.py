

# Original file: ./tf_efficientnet_b0___60.0/tf_efficientnet_b0___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/wj/cwjkujucocwfk2sc72xolmnajvekt6vmsgfpsdfzvzo7anth75ow.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_13 = async_compile.triton('triton_poi_fused_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_13', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_13(out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp7 = -9.502371540293097e-05
    tmp8 = 0.0029188317712396383
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = -0.0016967609990388155
    tmp11 = tl.where(tmp4, tmp10, tmp9)
    tmp12 = tl.full([1], 4, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.full([1], 5, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = 0.004726719111204147
    tmp17 = 0.0014568993356078863
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = 0.00952182523906231
    tmp20 = tl.where(tmp13, tmp19, tmp18)
    tmp21 = tl.where(tmp2, tmp11, tmp20)
    tl.store(out_ptr0 + (x0), tmp21, xmask)
''')
