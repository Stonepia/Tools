

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/sk/cskkwncnu4tv5knkusgfsarm7whyiqml6ghevbv3kp3mv3xl7qtb.py
# Source Nodes: [add, l__self___fpn_upsample], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# add => add_135
# l__self___fpn_upsample => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_130, add_132, add_133, add_134, convert_element_type_114, convert_element_type_116, iota, mul_171, mul_173, mul_175, mul_176, mul_177, mul_178, mul_179, mul_180, sub_57, sub_58, sub_59, sub_60
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_2 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 64)
    x1 = xindex % 64
    y0 = yindex
    x3 = xindex
    tmp41 = tl.load(in_ptr1 + (y0 + (256*x3)), ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49206349206349204
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (y0 + (256*tmp14) + (8192*tmp8)), ymask)
    tmp16 = tmp8.to(tl.float32)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp2 - tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.ceil(tmp7)
    tmp21 = 31.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22.to(tl.int32)
    tmp24 = tl.load(in_ptr0 + (y0 + (256*tmp14) + (8192*tmp23)), ymask)
    tmp25 = tmp24 * tmp17
    tmp26 = tmp19 + tmp25
    tmp27 = libdevice.ceil(tmp13)
    tmp28 = triton_helpers.minimum(tmp27, tmp21)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (y0 + (256*tmp29) + (8192*tmp8)), ymask)
    tmp31 = tmp30 * tmp18
    tmp32 = tl.load(in_ptr0 + (y0 + (256*tmp29) + (8192*tmp23)), ymask)
    tmp33 = tmp32 * tmp17
    tmp34 = tmp31 + tmp33
    tmp35 = tmp14.to(tl.float32)
    tmp36 = tmp13 - tmp35
    tmp37 = tmp2 - tmp36
    tmp38 = tmp26 * tmp37
    tmp39 = tmp34 * tmp36
    tmp40 = tmp38 + tmp39
    tmp42 = tmp40 + tmp41
    tl.store(out_ptr0 + (x3 + (4096*y0)), tmp26, ymask)
    tl.store(out_ptr1 + (x3 + (4096*y0)), tmp34, ymask)
    tl.store(out_ptr2 + (y0 + (256*x3)), tmp42, ymask)
''')
