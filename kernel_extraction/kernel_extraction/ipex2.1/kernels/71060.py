

# Original file: ./timm_nfnet___60.0/timm_nfnet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/yz/cyzk7kqfqcaz5h4ezpvoe773it7bkma46qx4dzyotgrwakltcwwd.py
# Source Nodes: [add_4, add_5, gelu_27, mul_44, mul_45, mul_46, mul_52, mul_53, mul_54, mul_55, mul__27, mul__32, mul__33], Original ATen: [aten.add, aten.gelu, aten.mul]
# add_4 => add_54
# add_5 => add_63
# gelu_27 => add_64, erf_27, mul_231, mul_232, mul_233
# mul_44 => mul_194
# mul_45 => mul_195
# mul_46 => mul_197
# mul_52 => mul_227
# mul_53 => mul_228
# mul_54 => mul_230
# mul_55 => mul_235
# mul__27 => mul_196
# mul__32 => mul_229
# mul__33 => mul_234
triton_poi_fused_add_gelu_mul_21 = async_compile.triton('triton_poi_fused_add_gelu_mul_21', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_gelu_mul_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28311552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 1536
    x2 = (xindex // 221184)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_out_ptr0 + (x3), None)
    tmp10 = tl.load(in_ptr2 + (x0 + (1536*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr3 + (x3), None)
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 * tmp5
    tmp7 = 0.2
    tmp8 = tmp6 * tmp7
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11 * tmp3
    tmp13 = tmp12 * tmp5
    tmp14 = tmp13 * tmp7
    tmp16 = tmp14 + tmp15
    tmp17 = tmp8 + tmp16
    tmp18 = 0.5
    tmp19 = tmp17 * tmp18
    tmp20 = 0.7071067811865476
    tmp21 = tmp17 * tmp20
    tmp22 = libdevice.erf(tmp21)
    tmp23 = 1.0
    tmp24 = tmp22 + tmp23
    tmp25 = tmp19 * tmp24
    tmp26 = 1.7015043497085571
    tmp27 = tmp25 * tmp26
    tmp28 = 0.9449111825230679
    tmp29 = tmp27 * tmp28
    tl.store(in_out_ptr0 + (x3), tmp17, None)
    tl.store(out_ptr0 + (x3), tmp29, None)
''')
