

# Original file: ./vision_maskrcnn__56_inference_96.36/vision_maskrcnn__56_inference_96.36_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/kp/ckpoqipdvnhd6peio2v2cyn7advauv3jn6vhbd5pg5u2yr7yyghb.py
# Source Nodes: [getitem_1, interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.select, aten.sub]
# getitem_1 => select_1
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_1, add_3, add_4, add_5, add_6, clamp_min_1, convert_element_type, convert_element_type_2, convert_element_type_5, iota_1, mul_1, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_1, sub_2, sub_3, sub_4, sub_5
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_select_sub_1 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_select_sub_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_select_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_select_sub_1(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 68)
    x0 = xindex % 68
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 0.29411764705882354
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 - tmp6
    tmp11 = triton_helpers.maximum(tmp10, tmp4)
    tmp12 = tmp11.to(tl.int32)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15 + tmp4
    tmp17 = tmp16 + tmp6
    tmp18 = 0.4411764705882353
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19 - tmp6
    tmp21 = triton_helpers.maximum(tmp20, tmp4)
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tl.load(in_ptr0 + (tmp22 + (30*tmp12)), xmask).to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp12.to(tl.float32)
    tmp26 = tmp11 - tmp25
    tmp27 = tmp2 - tmp26
    tmp28 = tmp24 * tmp27
    tmp29 = libdevice.ceil(tmp11)
    tmp30 = 29.0
    tmp31 = triton_helpers.minimum(tmp29, tmp30)
    tmp32 = tmp31.to(tl.int32)
    tmp33 = tl.load(in_ptr0 + (tmp22 + (30*tmp32)), xmask).to(tl.float32)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp26
    tmp36 = tmp28 + tmp35
    tmp37 = libdevice.ceil(tmp21)
    tmp38 = triton_helpers.minimum(tmp37, tmp30)
    tmp39 = tmp38.to(tl.int32)
    tmp40 = tl.load(in_ptr0 + (tmp39 + (30*tmp12)), xmask).to(tl.float32)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp27
    tmp43 = tl.load(in_ptr0 + (tmp39 + (30*tmp32)), xmask).to(tl.float32)
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp26
    tmp46 = tmp42 + tmp45
    tmp47 = tmp22.to(tl.float32)
    tmp48 = tmp21 - tmp47
    tmp49 = tmp2 - tmp48
    tmp50 = tmp36 * tmp49
    tmp51 = tmp46 * tmp48
    tmp52 = tmp50 + tmp51
    tmp53 = tmp52.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp53, xmask)
''')
