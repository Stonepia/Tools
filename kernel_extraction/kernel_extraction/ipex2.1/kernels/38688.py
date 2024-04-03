

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/qu/cqu4w6v3sjzmvjeto52xvztgddwe357mlzpfemdxi762vozplaiq.py
# Source Nodes: [interpolate_4], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate_4 => _unsafe_index_16, _unsafe_index_17, _unsafe_index_18, _unsafe_index_19, add_28, add_30, add_32, add_33, add_34, clamp_min_8, convert_element_type_24, convert_element_type_26, iota_8, mul_60, mul_62, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, sub_24, sub_26, sub_27, sub_28, sub_29
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_15 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512, 131072], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_15(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 123904
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 352)
    x2 = xindex % 352
    y0 = yindex % 64
    y1 = (yindex // 64)
    x4 = xindex
    y5 = yindex
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = tmp8 - tmp6
    tmp10 = triton_helpers.maximum(tmp9, tmp4)
    tmp11 = tmp10.to(tl.int32)
    tmp12 = x2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp2
    tmp15 = tmp14 + tmp4
    tmp16 = tmp15 + tmp6
    tmp17 = tmp16 * tmp6
    tmp18 = tmp17 - tmp6
    tmp19 = triton_helpers.maximum(tmp18, tmp4)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.load(in_ptr0 + (y0 + (64*tmp20) + (11264*tmp11) + (1982464*y1)), xmask & ymask)
    tmp22 = tmp11.to(tl.float32)
    tmp23 = tmp10 - tmp22
    tmp24 = tmp2 - tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = libdevice.ceil(tmp10)
    tmp27 = 175.0
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (y0 + (64*tmp20) + (11264*tmp29) + (1982464*y1)), xmask & ymask)
    tmp31 = tmp30 * tmp23
    tmp32 = tmp25 + tmp31
    tmp33 = libdevice.ceil(tmp19)
    tmp34 = triton_helpers.minimum(tmp33, tmp27)
    tmp35 = tmp34.to(tl.int32)
    tmp36 = tl.load(in_ptr0 + (y0 + (64*tmp35) + (11264*tmp11) + (1982464*y1)), xmask & ymask)
    tmp37 = tmp36 * tmp24
    tmp38 = tl.load(in_ptr0 + (y0 + (64*tmp35) + (11264*tmp29) + (1982464*y1)), xmask & ymask)
    tmp39 = tmp38 * tmp23
    tmp40 = tmp37 + tmp39
    tmp41 = tmp20.to(tl.float32)
    tmp42 = tmp19 - tmp41
    tmp43 = tmp2 - tmp42
    tmp44 = tmp32 * tmp43
    tmp45 = tmp40 * tmp42
    tmp46 = tmp44 + tmp45
    tl.store(out_ptr2 + (y0 + (64*x4) + (7929856*y1)), tmp46, xmask & ymask)
''')