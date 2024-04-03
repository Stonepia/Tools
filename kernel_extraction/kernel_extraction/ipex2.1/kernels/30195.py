

# Original file: ./vision_maskrcnn__58_inference_98.38/vision_maskrcnn__58_inference_98.38.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/pn/cpnfyjqigxvarfrcl2vrpvlq6gpek2lmk3vreabnkif6qvr6ki4j.py
# Source Nodes: [interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_1, add_3, add_4, add_5, add_6, clamp_min_1, convert_element_type_1, convert_element_type_4, iota_1, mul_1, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_3, sub_4, sub_5, sub_6, sub_7
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_1 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_1(in_out_ptr0, in_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
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
    tmp25 = tl.load(in_ptr0 + (tmp24 + (30*tmp13)), xmask)
    tmp26 = tmp13.to(tl.float32)
    tmp27 = tmp12 - tmp26
    tmp28 = tmp2 - tmp27
    tmp29 = tmp25 * tmp28
    tmp30 = libdevice.ceil(tmp12)
    tmp31 = 29.0
    tmp32 = triton_helpers.minimum(tmp30, tmp31)
    tmp33 = tmp32.to(tl.int64)
    tmp34 = tl.load(in_ptr0 + (tmp24 + (30*tmp33)), xmask)
    tmp35 = tmp34 * tmp27
    tmp36 = tmp29 + tmp35
    tmp37 = libdevice.ceil(tmp23)
    tmp38 = triton_helpers.minimum(tmp37, tmp31)
    tmp39 = tmp38.to(tl.int64)
    tmp40 = tl.load(in_ptr0 + (tmp39 + (30*tmp13)), xmask)
    tmp41 = tmp40 * tmp28
    tmp42 = tmp32.to(tl.int32)
    tmp43 = tmp38.to(tl.int32)
    tmp44 = tl.load(in_ptr0 + (tmp43 + (30*tmp42)), xmask)
    tmp45 = tmp44 * tmp27
    tmp46 = tmp41 + tmp45
    tmp47 = tmp24.to(tl.float32)
    tmp48 = tmp23 - tmp47
    tmp49 = tmp2 - tmp48
    tmp50 = tmp36 * tmp49
    tmp51 = tmp46 * tmp48
    tmp52 = tmp50 + tmp51
    tl.store(in_out_ptr0 + (x2), tmp52, xmask)
''')
