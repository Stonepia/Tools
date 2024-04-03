

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6l/c6lvt553xjlbz3iraiknbpi5cdqpqldmzsmrwovgf74ufxdvgnuo.py
# Source Nodes: [interpolate_3], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate_3 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_21, add_23, add_25, add_26, add_27, clamp_min_6, convert_element_type_18, convert_element_type_20, iota_6, mul_48, mul_50, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, sub_18, sub_20, sub_21, sub_22, sub_23
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_13 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_13', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1024, 32768], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_13(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 30976
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 176)
    x2 = xindex % 176
    y0 = yindex % 128
    y1 = (yindex // 128)
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
    tmp21 = tl.load(in_ptr0 + (y0 + (128*tmp20) + (11264*tmp11) + (991232*y1)), xmask & ymask)
    tmp22 = tmp11.to(tl.float32)
    tmp23 = tmp10 - tmp22
    tmp24 = tmp2 - tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = libdevice.ceil(tmp10)
    tmp27 = 87.0
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (y0 + (128*tmp20) + (11264*tmp29) + (991232*y1)), xmask & ymask)
    tmp31 = tmp30 * tmp23
    tmp32 = tmp25 + tmp31
    tmp33 = libdevice.ceil(tmp19)
    tmp34 = triton_helpers.minimum(tmp33, tmp27)
    tmp35 = tmp34.to(tl.int32)
    tmp36 = tl.load(in_ptr0 + (y0 + (128*tmp35) + (11264*tmp11) + (991232*y1)), xmask & ymask)
    tmp37 = tmp36 * tmp24
    tmp38 = tl.load(in_ptr0 + (y0 + (128*tmp35) + (11264*tmp29) + (991232*y1)), xmask & ymask)
    tmp39 = tmp38 * tmp23
    tmp40 = tmp37 + tmp39
    tmp41 = tmp20.to(tl.float32)
    tmp42 = tmp19 - tmp41
    tmp43 = tmp2 - tmp42
    tmp44 = tmp32 * tmp43
    tmp45 = tmp40 * tmp42
    tmp46 = tmp44 + tmp45
    tl.store(out_ptr2 + (y0 + (128*x4) + (3964928*y1)), tmp46, xmask & ymask)
''')
