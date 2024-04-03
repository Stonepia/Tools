

# Original file: ./vision_maskrcnn__56_inference_96.36/vision_maskrcnn__56_inference_96.36.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/zy/czyqe5766civ4hfzgqcbexmsblon6ubzojdb7bdhdp3retu5bt3p.py
# Source Nodes: [interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_1, add_3, add_4, add_5, add_6, clamp_min_1, convert_element_type_1, convert_element_type_4, iota_1, mul_1, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, sub_1, sub_2, sub_3, sub_4, sub_5
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

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25370
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 118)
    x0 = xindex % 118
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = 0.13953488372093023
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9 - tmp6
    tmp11 = triton_helpers.maximum(tmp10, tmp4)
    tmp12 = tmp11.to(tl.int32)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp2
    tmp16 = tmp15 + tmp4
    tmp17 = tmp16 + tmp6
    tmp18 = 0.2542372881355932
    tmp19 = tmp17 * tmp18
    tmp20 = tmp19 - tmp6
    tmp21 = triton_helpers.maximum(tmp20, tmp4)
    tmp22 = tmp21.to(tl.int32)
    tmp23 = tl.load(in_ptr0 + (tmp22 + (30*tmp12)), xmask)
    tmp24 = tmp12.to(tl.float32)
    tmp25 = tmp11 - tmp24
    tmp26 = tmp2 - tmp25
    tmp27 = tmp23 * tmp26
    tmp28 = libdevice.ceil(tmp11)
    tmp29 = 29.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp22 + (30*tmp31)), xmask)
    tmp33 = tmp32 * tmp25
    tmp34 = tmp27 + tmp33
    tmp35 = libdevice.ceil(tmp21)
    tmp36 = triton_helpers.minimum(tmp35, tmp29)
    tmp37 = tmp36.to(tl.int32)
    tmp38 = tl.load(in_ptr0 + (tmp37 + (30*tmp12)), xmask)
    tmp39 = tmp38 * tmp26
    tmp40 = tl.load(in_ptr0 + (tmp37 + (30*tmp31)), xmask)
    tmp41 = tmp40 * tmp25
    tmp42 = tmp39 + tmp41
    tmp43 = tmp22.to(tl.float32)
    tmp44 = tmp21 - tmp43
    tmp45 = tmp2 - tmp44
    tmp46 = tmp34 * tmp45
    tmp47 = tmp42 * tmp44
    tmp48 = tmp46 + tmp47
    tl.store(in_out_ptr0 + (x2), tmp48, xmask)
''')
