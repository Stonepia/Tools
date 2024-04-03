

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/cy/ccyd3asubt6sxlek2ieueaetc4nw5ovao5jeaeel6fi4gw2k4w4r.py
# Source Nodes: [add_1, add_2, l__self___fpn_upsample_1, l__self___fpn_upsample_2], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# add_1 => add_141
# add_2 => add_147
# l__self___fpn_upsample_1 => add_140
# l__self___fpn_upsample_2 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_142, add_144, add_145, add_146, convert_element_type_126, convert_element_type_128, iota_4, mul_191, mul_193, mul_195, mul_196, mul_197, mul_198, mul_199, mul_200, sub_65, sub_66, sub_67, sub_68
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[256, 65536], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_4(in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 65536
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 256)
    x1 = xindex % 256
    y0 = yindex
    x3 = xindex
    tmp57 = tl.load(in_out_ptr1 + (y0 + (256*x3)), ymask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.4980392156862745
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (tmp14 + (128*tmp8) + (16384*y0)), ymask)
    tmp16 = tl.load(in_ptr1 + (tmp14 + (128*tmp8) + (16384*y0)), ymask)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.load(in_ptr2 + (y0 + (256*tmp14) + (32768*tmp8)), ymask)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp8.to(tl.float32)
    tmp21 = tmp7 - tmp20
    tmp22 = tmp2 - tmp21
    tmp23 = tmp19 * tmp22
    tmp24 = libdevice.ceil(tmp7)
    tmp25 = 127.0
    tmp26 = triton_helpers.minimum(tmp24, tmp25)
    tmp27 = tmp26.to(tl.int32)
    tmp28 = tl.load(in_ptr0 + (tmp14 + (128*tmp27) + (16384*y0)), ymask)
    tmp29 = tl.load(in_ptr1 + (tmp14 + (128*tmp27) + (16384*y0)), ymask)
    tmp30 = tmp28 + tmp29
    tmp31 = tl.load(in_ptr2 + (y0 + (256*tmp14) + (32768*tmp27)), ymask)
    tmp32 = tmp30 + tmp31
    tmp33 = tmp32 * tmp21
    tmp34 = tmp23 + tmp33
    tmp35 = libdevice.ceil(tmp13)
    tmp36 = triton_helpers.minimum(tmp35, tmp25)
    tmp37 = tmp36.to(tl.int32)
    tmp38 = tl.load(in_ptr0 + (tmp37 + (128*tmp27) + (16384*y0)), ymask)
    tmp39 = tl.load(in_ptr1 + (tmp37 + (128*tmp27) + (16384*y0)), ymask)
    tmp40 = tmp38 + tmp39
    tmp41 = tl.load(in_ptr2 + (y0 + (256*tmp37) + (32768*tmp27)), ymask)
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42 * tmp21
    tmp44 = tl.load(in_ptr0 + (tmp37 + (128*tmp8) + (16384*y0)), ymask)
    tmp45 = tl.load(in_ptr1 + (tmp37 + (128*tmp8) + (16384*y0)), ymask)
    tmp46 = tmp44 + tmp45
    tmp47 = tl.load(in_ptr2 + (y0 + (256*tmp37) + (32768*tmp8)), ymask)
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48 * tmp22
    tmp50 = tmp49 + tmp43
    tmp51 = tmp14.to(tl.float32)
    tmp52 = tmp13 - tmp51
    tmp53 = tmp2 - tmp52
    tmp54 = tmp34 * tmp53
    tmp55 = tmp50 * tmp52
    tmp56 = tmp54 + tmp55
    tmp58 = tmp56 + tmp57
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (y0 + (256*x3)), tmp58, ymask)
''')
