

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/gm/cgmxrgoyu4c47cckgvwovmqn6fmjeujo5l4xbrvouewnj2b3v2k2.py
# Source Nodes: [l__self___fpn_out_branches_2_3, l__self___fpn_upsample_2], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# l__self___fpn_out_branches_2_3 => _unsafe_index_20, _unsafe_index_21, _unsafe_index_22, _unsafe_index_23, add_166, add_167, add_168, convert_element_type_281, convert_element_type_284, convert_element_type_288, mul_232, mul_234, mul_235, mul_236, mul_237, mul_238, mul_239, sub_80, sub_81, sub_82, sub_83
# l__self___fpn_upsample_2 => add_142, convert_element_type_246, iota_4, mul_191
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7(in_ptr0, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 256
    x0 = xindex % 256
    x2 = (xindex // 65536)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.24705882352941178
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (x2 + (64*tmp14) + (4096*tmp8)), None).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = libdevice.ceil(tmp7)
    tmp22 = 63.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (x2 + (64*tmp14) + (4096*tmp24)), None).to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 * tmp18
    tmp28 = tmp20 + tmp27
    tmp29 = libdevice.ceil(tmp13)
    tmp30 = triton_helpers.minimum(tmp29, tmp22)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (x2 + (64*tmp31) + (4096*tmp8)), None).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp19
    tmp35 = tl.load(in_ptr0 + (x2 + (64*tmp31) + (4096*tmp24)), None).to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp36 * tmp18
    tmp38 = tmp34 + tmp37
    tmp39 = tmp14.to(tl.float32)
    tmp40 = tmp13 - tmp39
    tmp41 = tmp2 - tmp40
    tmp42 = tmp28 * tmp41
    tmp43 = tmp38 * tmp40
    tmp44 = tmp42 + tmp43
    tmp45 = tmp44.to(tl.float32)
    tl.store(out_ptr2 + (x4), tmp45, None)
''')