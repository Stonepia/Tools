

# Original file: ./demucs__23_inference_63.3/demucs__23_inference_63.3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ni/cni6baj7tzr5bandvyeiak6mciv7cmmuujy7f2exavbyeowxhtp3.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_12 = async_compile.triton('triton_poi_fused_12', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused_12(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23767808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 371372) % 8
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = x1
    tmp2 = tl.full([1], 4, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp1 < tmp6
    tmp8 = 0.07501220703125
    tmp9 = 0.0155181884765625
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tl.full([1], 3, tl.int64)
    tmp12 = tmp1 < tmp11
    tmp13 = -0.12176513671875
    tmp14 = 0.0213470458984375
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.where(tmp5, tmp10, tmp15)
    tmp17 = tl.full([1], 6, tl.int64)
    tmp18 = tmp1 < tmp17
    tmp19 = tl.full([1], 5, tl.int64)
    tmp20 = tmp1 < tmp19
    tmp21 = -0.10577392578125
    tmp22 = 0.0071868896484375
    tmp23 = tl.where(tmp20, tmp21, tmp22)
    tmp24 = tl.full([1], 7, tl.int64)
    tmp25 = tmp1 < tmp24
    tmp26 = -0.099853515625
    tmp27 = 0.1341552734375
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tl.where(tmp18, tmp23, tmp28)
    tmp30 = tl.where(tmp3, tmp16, tmp29)
    tmp31 = tmp0 + tmp30
    tl.store(in_out_ptr0 + (x3), tmp31, xmask)
''')
