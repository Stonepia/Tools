

# Original file: ./vision_maskrcnn__58_inference_98.38/vision_maskrcnn__58_inference_98.38_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ya/cyadassjbvehevvsun5mbdbx3wxii7ew7ybjb4lvc7f7ptprgyxf.py
# Source Nodes: [getitem_1, interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.select, aten.sub]
# getitem_1 => select_1
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_1, add_3, add_4, add_5, add_6, clamp_min_1, convert_element_type, convert_element_type_2, convert_element_type_5, iota_1, mul_1, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_3, sub_4, sub_5, sub_6, sub_7
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

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_select_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_select_sub_1(in_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // ks0)
    x0 = xindex % ks0
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 30*(1/ks1)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp10 - tmp6
    tmp12 = triton_helpers.maximum(tmp11, tmp4)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = x0
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp2
    tmp17 = tmp16 + tmp4
    tmp18 = tmp17 + tmp6
    tmp19 = 30*(1/ks0)
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = tmp21 - tmp6
    tmp23 = triton_helpers.maximum(tmp22, tmp4)
    tmp24 = tmp23.to(tl.int64)
    tmp25 = tl.load(in_ptr0 + (tmp24 + (30*tmp13)), xmask).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp13.to(tl.float32)
    tmp28 = tmp12 - tmp27
    tmp29 = tmp2 - tmp28
    tmp30 = tmp26 * tmp29
    tmp31 = libdevice.ceil(tmp12)
    tmp32 = 29.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = tmp33.to(tl.int64)
    tmp35 = tl.load(in_ptr0 + (tmp24 + (30*tmp34)), xmask).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp28
    tmp38 = tmp30 + tmp37
    tmp39 = libdevice.ceil(tmp23)
    tmp40 = triton_helpers.minimum(tmp39, tmp32)
    tmp41 = tmp40.to(tl.int64)
    tmp42 = tl.load(in_ptr0 + (tmp41 + (30*tmp13)), xmask).to(tl.float32)
    tmp43 = tmp42.to(tl.float32)
    tmp44 = tmp43 * tmp29
    tmp45 = tmp33.to(tl.int32)
    tmp46 = tmp40.to(tl.int32)
    tmp47 = tl.load(in_ptr0 + (tmp46 + (30*tmp45)), xmask).to(tl.float32)
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 * tmp28
    tmp50 = tmp44 + tmp49
    tmp51 = tmp24.to(tl.float32)
    tmp52 = tmp23 - tmp51
    tmp53 = tmp2 - tmp52
    tmp54 = tmp38 * tmp53
    tmp55 = tmp50 * tmp52
    tmp56 = tmp54 + tmp55
    tmp57 = tmp56.to(tl.float32)
    tl.store(out_ptr1 + (x2), tmp57, xmask)
''')
