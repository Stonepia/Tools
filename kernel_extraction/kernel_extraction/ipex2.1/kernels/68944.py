

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/fp/cfpqwyzsvsio5uqrp7fjbiw2krqgmd4dxfpvbvhcnbdq4sobbepl.py
# Source Nodes: [l__self___fpn_out_branches_2_3, l__self___fpn_upsample_2], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# l__self___fpn_out_branches_2_3 => _unsafe_index_20, _unsafe_index_21, _unsafe_index_22, _unsafe_index_23, add_166, add_167, add_168, convert_element_type_152, mul_232, mul_234, mul_235, mul_236, mul_237, mul_238, mul_239, sub_80, sub_81, sub_82, sub_83
# l__self___fpn_upsample_2 => add_142, convert_element_type_126, iota_4, mul_191
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

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
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
    tmp15 = tl.load(in_ptr0 + (x2 + (64*tmp14) + (4096*tmp8)), None)
    tmp16 = tmp8.to(tl.float32)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp2 - tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.ceil(tmp7)
    tmp21 = 63.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22.to(tl.int32)
    tmp24 = tl.load(in_ptr0 + (x2 + (64*tmp14) + (4096*tmp23)), None)
    tmp25 = tmp24 * tmp17
    tmp26 = tmp19 + tmp25
    tmp27 = libdevice.ceil(tmp13)
    tmp28 = triton_helpers.minimum(tmp27, tmp21)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (x2 + (64*tmp29) + (4096*tmp8)), None)
    tmp31 = tmp30 * tmp18
    tmp32 = tl.load(in_ptr0 + (x2 + (64*tmp29) + (4096*tmp23)), None)
    tmp33 = tmp32 * tmp17
    tmp34 = tmp31 + tmp33
    tmp35 = tmp14.to(tl.float32)
    tmp36 = tmp13 - tmp35
    tmp37 = tmp2 - tmp36
    tmp38 = tmp26 * tmp37
    tmp39 = tmp34 * tmp36
    tmp40 = tmp38 + tmp39
    tl.store(out_ptr2 + (x4), tmp40, None)
''')
