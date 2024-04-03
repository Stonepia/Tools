

# Original file: ./doctr_det_predictor___60.0/doctr_det_predictor___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/66/c66q64voqijx6i6otu7b5sb3qkh2dh43e7dlly4c7krefkvg2p26.py
# Source Nodes: [add_2, l__self___fpn_upsample_2], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# add_2 => add_147
# l__self___fpn_upsample_2 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_142, add_144, add_145, add_146, convert_element_type_246, convert_element_type_248, convert_element_type_252, iota_4, mul_191, mul_193, mul_195, mul_196, mul_197, mul_198, mul_199, mul_200, sub_65, sub_66, sub_67, sub_68
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

@pointwise(size_hints=[256, 65536], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_4(in_out_ptr0, in_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp42 = tl.load(in_out_ptr0 + (y0 + (256*x3)), ymask, eviction_policy='evict_last').to(tl.float32)
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
    tmp16 = tmp8.to(tl.float32)
    tmp17 = tmp7 - tmp16
    tmp18 = tmp2 - tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = libdevice.ceil(tmp7)
    tmp21 = 127.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22.to(tl.int32)
    tmp24 = tl.load(in_ptr0 + (tmp14 + (128*tmp23) + (16384*y0)), ymask)
    tmp25 = tmp24 * tmp17
    tmp26 = tmp19 + tmp25
    tmp27 = libdevice.ceil(tmp13)
    tmp28 = triton_helpers.minimum(tmp27, tmp21)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (tmp29 + (128*tmp8) + (16384*y0)), ymask)
    tmp31 = tmp30 * tmp18
    tmp32 = tl.load(in_ptr0 + (tmp29 + (128*tmp23) + (16384*y0)), ymask)
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (y0 + (256*x3)), tmp43, ymask)
''')
