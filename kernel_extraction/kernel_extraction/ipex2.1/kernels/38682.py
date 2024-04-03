

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/e2/ce22ndtibmzf4bpumpuup4kagtx4t7cgakjc465bw2g4cuw6ndyc.py
# Source Nodes: [interpolate_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.clamp, aten.mul, aten.rsub, aten.sub]
# interpolate_1 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_11, add_12, add_13, add_7, add_9, clamp_min_2, convert_element_type_6, convert_element_type_8, iota_2, mul_24, mul_26, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, sub_10, sub_11, sub_6, sub_8, sub_9
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_9 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_9', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_9(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 1936
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 44)
    x2 = xindex % 44
    y0 = yindex % 512
    y1 = (yindex // 512)
    x4 = xindex
    y5 = yindex
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 * tmp6
    tmp9 = tmp8 - tmp6
    tmp10 = triton_helpers.maximum(tmp9, tmp4)
    tmp11 = tmp10.to(tl.int32)
    tmp12 = x2
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp2
    tmp15 = tmp14 + tmp4
    tmp16 = tmp15 + tmp6
    tmp17 = tmp16 * tmp6
    tmp18 = tmp17 - tmp6
    tmp19 = triton_helpers.maximum(tmp18, tmp4)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.load(in_ptr0 + (y0 + (512*tmp20) + (11264*tmp11) + (247808*y1)), xmask)
    tmp22 = tmp11.to(tl.float32)
    tmp23 = tmp10 - tmp22
    tmp24 = tmp2 - tmp23
    tmp25 = tmp21 * tmp24
    tmp26 = libdevice.ceil(tmp10)
    tmp27 = 21.0
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tmp29 = tmp28.to(tl.int32)
    tmp30 = tl.load(in_ptr0 + (y0 + (512*tmp20) + (11264*tmp29) + (247808*y1)), xmask)
    tmp31 = tmp30 * tmp23
    tmp32 = tmp25 + tmp31
    tmp33 = libdevice.ceil(tmp19)
    tmp34 = triton_helpers.minimum(tmp33, tmp27)
    tmp35 = tmp34.to(tl.int32)
    tmp36 = tl.load(in_ptr0 + (y0 + (512*tmp35) + (11264*tmp11) + (247808*y1)), xmask)
    tmp37 = tmp36 * tmp24
    tmp38 = tl.load(in_ptr0 + (y0 + (512*tmp35) + (11264*tmp29) + (247808*y1)), xmask)
    tmp39 = tmp38 * tmp23
    tmp40 = tmp37 + tmp39
    tmp41 = tmp20.to(tl.float32)
    tmp42 = tmp19 - tmp41
    tmp43 = tmp2 - tmp42
    tmp44 = tmp32 * tmp43
    tmp45 = tmp40 * tmp42
    tmp46 = tmp44 + tmp45
    tl.store(out_ptr2 + (y0 + (512*x4) + (991232*y1)), tmp46, xmask)
''')
