

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/n2/cn27nmrcjd4xxu55h2huxw2gr32fbti5xgqhuq57bvlrdelvnrgo.py
# Source Nodes: [add_1, l__self___fpn_upsample_1, l__self___fpn_upsample_2], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# add_1 => add_141
# l__self___fpn_upsample_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_136, add_138, add_139, add_140, convert_element_type_238, convert_element_type_240, convert_element_type_244, iota_2, mul_181, mul_183, mul_185, mul_186, mul_187, mul_188, mul_189, mul_190, sub_61, sub_62, sub_63, sub_64
# l__self___fpn_upsample_2 => convert_element_type_245
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_3(in_ptr0, in_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 16384
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 128)
    x1 = xindex % 128
    y0 = yindex
    x3 = xindex
    tmp42 = tl.load(in_ptr1 + (y0 + (256*x3)), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49606299212598426
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (tmp14 + (64*tmp8) + (4096*y0)), ymask)
    tmp16 = tmp8.to(tl.float32)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp2 - tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.ceil(tmp7)
    tmp21 = 63.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22.to(tl.int32)
    tmp24 = tl.load(in_ptr0 + (tmp14 + (64*tmp23) + (4096*y0)), ymask)
    tmp25 = tmp24 * tmp17
    tmp26 = tmp19 + tmp25
    tmp27 = libdevice.ceil(tmp13)
    tmp28 = triton_helpers.minimum(tmp27, tmp21)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (tmp29 + (64*tmp8) + (4096*y0)), ymask)
    tmp31 = tmp30 * tmp18
    tmp32 = tl.load(in_ptr0 + (tmp29 + (64*tmp23) + (4096*y0)), ymask)
    tmp33 = tmp32 * tmp17
    tmp34 = tmp31 + tmp33
    tmp35 = tmp14.to(tl.float32)
    tmp36 = tmp13 - tmp35
    tmp37 = tmp2 - tmp36
    tmp38 = tmp26 * tmp37
    tmp39 = tmp34 * tmp36
    tmp40 = tmp38 + tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp43 = tmp41 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr2 + (x3 + (16384*y0)), tmp44, ymask)
    tl.store(out_ptr3 + (y0 + (256*x3)), tmp43, ymask)
''')
