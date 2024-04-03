

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/d6/cd6mqtd4if5sr44j2vbw25aetmqirqdgvldafrpkvklztcnrxhsb.py
# Source Nodes: [add_13, mul_15, mul_17, sigmoid, sub_7], Original ATen: [aten.add, aten.mul, aten.rsub, aten.sigmoid]
# add_13 => add_113
# mul_15 => mul_205
# mul_17 => mul_207
# sigmoid => sigmoid
# sub_7 => sub_85
triton_poi_fused_add_mul_rsub_sigmoid_41 = async_compile.triton('triton_poi_fused_add_mul_rsub_sigmoid_41', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_rsub_sigmoid_41', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_mul_rsub_sigmoid_41(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 743424
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 123904)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (4 + (5*x2)), None).to(tl.float32)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 7, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 7), "index out of bounds: 0 <= tmp1 < 7")
    tmp2 = tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 3.5
    tmp5 = tmp3 < tmp4
    tmp6 = 0.125
    tmp7 = tmp3 * tmp6
    tmp8 = tmp7 + tmp6
    tmp9 = 6 + ((-1)*tmp1)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp6
    tmp12 = 0.875
    tmp13 = tmp12 - tmp11
    tmp14 = tl.where(tmp5, tmp8, tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 0.0
    tmp20 = tmp18 > tmp19
    tmp21 = 0.1
    tmp22 = tmp18 * tmp21
    tmp23 = tl.where(tmp20, tmp18, tmp22)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.sigmoid(tmp24)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp16 * tmp26
    tmp28 = tmp15 - tmp25
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp14 * tmp29
    tmp31 = tmp27 + tmp30
    tl.store(out_ptr0 + (x2), tmp31, None)
''')
