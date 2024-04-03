

# Original file: ./pnasnet5large___60.0/pnasnet5large___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/pq/cpqwuo5sgqj3x7gwe56pahw5nwf74tzeeujtxcvhwjorn7y72p3s.py
# Source Nodes: [max_pool2d_3, pad_10], Original ATen: [aten.constant_pad_nd, aten.max_pool2d_with_indices]
# max_pool2d_3 => max_pool2d_with_indices_3
# pad_10 => constant_pad_nd_11
triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_9 = async_compile.triton('triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_constant_pad_nd_max_pool2d_with_indices_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3048192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4536) % 42
    x1 = (xindex // 108) % 42
    x0 = xindex % 108
    x3 = (xindex // 190512)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 83, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = (-1) + (2*x1)
    tmp6 = tmp5 >= tmp1
    tmp7 = tmp5 < tmp3
    tmp8 = tmp2 & tmp4
    tmp9 = tmp8 & tmp6
    tmp10 = tmp9 & tmp7
    tmp11 = tl.load(in_ptr0 + ((-9072) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp10 & xmask, other=0.0)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 2*x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp8 & tmp14
    tmp17 = tmp16 & tmp15
    tmp18 = tl.load(in_ptr0 + ((-8964) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp17 & xmask, other=0.0)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp12)
    tmp21 = 1 + (2*x1)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp8 & tmp22
    tmp25 = tmp24 & tmp23
    tmp26 = tl.load(in_ptr0 + ((-8856) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp25 & xmask, other=0.0)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = triton_helpers.maximum(tmp27, tmp20)
    tmp29 = 2*x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp6
    tmp34 = tmp33 & tmp7
    tmp35 = tl.load(in_ptr0 + ((-108) + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp34 & xmask, other=0.0)
    tmp36 = tl.where(tmp34, tmp35, float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp28)
    tmp38 = tmp32 & tmp14
    tmp39 = tmp38 & tmp15
    tmp40 = tl.load(in_ptr0 + (x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp39 & xmask, other=0.0)
    tmp41 = tl.where(tmp39, tmp40, float("-inf"))
    tmp42 = triton_helpers.maximum(tmp41, tmp37)
    tmp43 = tmp32 & tmp22
    tmp44 = tmp43 & tmp23
    tmp45 = tl.load(in_ptr0 + (108 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp44 & xmask, other=0.0)
    tmp46 = tl.where(tmp44, tmp45, float("-inf"))
    tmp47 = triton_helpers.maximum(tmp46, tmp42)
    tmp48 = 1 + (2*x2)
    tmp49 = tmp48 >= tmp1
    tmp50 = tmp48 < tmp3
    tmp51 = tmp49 & tmp50
    tmp52 = tmp51 & tmp6
    tmp53 = tmp52 & tmp7
    tmp54 = tl.load(in_ptr0 + (8856 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp53 & xmask, other=0.0)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp47)
    tmp57 = tmp51 & tmp14
    tmp58 = tmp57 & tmp15
    tmp59 = tl.load(in_ptr0 + (8964 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp58 & xmask, other=0.0)
    tmp60 = tl.where(tmp58, tmp59, float("-inf"))
    tmp61 = triton_helpers.maximum(tmp60, tmp56)
    tmp62 = tmp51 & tmp22
    tmp63 = tmp62 & tmp23
    tmp64 = tl.load(in_ptr0 + (9072 + x0 + (216*x1) + (17928*x2) + (744012*x3)), tmp63 & xmask, other=0.0)
    tmp65 = tl.where(tmp63, tmp64, float("-inf"))
    tmp66 = triton_helpers.maximum(tmp65, tmp61)
    tl.store(out_ptr0 + (x6), tmp66, xmask)
''')
