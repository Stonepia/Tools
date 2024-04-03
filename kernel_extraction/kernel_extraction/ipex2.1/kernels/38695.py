

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/6c/c6cbaw2xa663h6m35c57t7laqb7bff2dy3atkqwi5xuumzo5krm2.py
# Source Nodes: [grid_sample_1], Original ATen: [aten.grid_sampler_2d]
# grid_sample_1 => add_53, add_54, add_55, index_10, index_11, index_12, index_13, mul_102, mul_103, mul_104, mul_105
triton_poi_fused_grid_sampler_2d_22 = async_compile.triton('triton_poi_fused_grid_sampler_2d_22', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32, 131072], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_grid_sampler_2d_22', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 10), equal_to_1=())]})
@triton.jit
def triton_poi_fused_grid_sampler_2d_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 18
    xnumel = 123904
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = (yindex // 3)
    y3 = yindex
    y0 = yindex % 3
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (247808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + (2*x2) + (247808*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr2 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr3 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr4 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr5 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr6 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr7 + (x2 + (123904*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 176.0
    tmp2 = tmp0 * tmp1
    tmp3 = 175.5
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 352.0
    tmp9 = tmp5 < tmp8
    tmp11 = tmp10 * tmp1
    tmp12 = tmp11 + tmp3
    tmp13 = libdevice.floor(tmp12)
    tmp14 = tmp13 >= tmp6
    tmp15 = tmp13 < tmp8
    tmp16 = tmp14 & tmp15
    tmp17 = tmp9 & tmp16
    tmp18 = tmp7 & tmp17
    tmp19 = tmp13.to(tl.int64)
    tmp20 = tl.full([1, 1], 0, tl.int64)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp21 < 0, tmp21 + 352, tmp21)
    # tl.device_assert((0 <= tmp22) & (tmp22 < 352), "index out of bounds: 0 <= tmp22 < 352")
    tmp23 = tmp5.to(tl.int64)
    tmp24 = tl.where(tmp18, tmp23, tmp20)
    tmp25 = tl.where(tmp24 < 0, tmp24 + 352, tmp24)
    # tl.device_assert((0 <= tmp25) & (tmp25 < 352), "index out of bounds: 0 <= tmp25 < 352")
    tmp26 = tl.load(in_ptr1 + (tmp25 + (352*tmp22) + (123904*y3)), ymask)
    tmp27 = 1.0
    tmp28 = tmp5 + tmp27
    tmp29 = tmp28 >= tmp6
    tmp30 = tmp28 < tmp8
    tmp31 = tmp30 & tmp16
    tmp32 = tmp29 & tmp31
    tmp33 = tl.where(tmp32, tmp19, tmp20)
    tmp34 = tl.where(tmp33 < 0, tmp33 + 352, tmp33)
    # tl.device_assert((0 <= tmp34) & (tmp34 < 352), "index out of bounds: 0 <= tmp34 < 352")
    tmp35 = tmp28.to(tl.int64)
    tmp36 = tl.where(tmp32, tmp35, tmp20)
    tmp37 = tl.where(tmp36 < 0, tmp36 + 352, tmp36)
    # tl.device_assert((0 <= tmp37) & (tmp37 < 352), "index out of bounds: 0 <= tmp37 < 352")
    tmp38 = tl.load(in_ptr1 + (tmp37 + (352*tmp34) + (123904*y3)), ymask)
    tmp39 = tmp13 + tmp27
    tmp40 = tmp39 >= tmp6
    tmp41 = tmp39 < tmp8
    tmp42 = tmp40 & tmp41
    tmp43 = tmp9 & tmp42
    tmp44 = tmp7 & tmp43
    tmp45 = tmp39.to(tl.int64)
    tmp46 = tl.where(tmp44, tmp45, tmp20)
    tmp47 = tl.where(tmp46 < 0, tmp46 + 352, tmp46)
    # tl.device_assert((0 <= tmp47) & (tmp47 < 352), "index out of bounds: 0 <= tmp47 < 352")
    tmp48 = tl.where(tmp44, tmp23, tmp20)
    tmp49 = tl.where(tmp48 < 0, tmp48 + 352, tmp48)
    # tl.device_assert((0 <= tmp49) & (tmp49 < 352), "index out of bounds: 0 <= tmp49 < 352")
    tmp50 = tl.load(in_ptr1 + (tmp49 + (352*tmp47) + (123904*y3)), ymask)
    tmp52 = tmp26 * tmp51
    tmp54 = tmp38 * tmp53
    tmp55 = tmp52 + tmp54
    tmp57 = tmp50 * tmp56
    tmp58 = tmp55 + tmp57
    tmp60 = tl.where(tmp59 < 0, tmp59 + 352, tmp59)
    # tl.device_assert(((0 <= tmp60) & (tmp60 < 352)) | ~(ymask & xmask), "index out of bounds: 0 <= tmp60 < 352")
    tmp62 = tl.where(tmp61 < 0, tmp61 + 352, tmp61)
    # tl.device_assert(((0 <= tmp62) & (tmp62 < 352)) | ~(ymask & xmask), "index out of bounds: 0 <= tmp62 < 352")
    tmp63 = tl.load(in_ptr1 + (tmp62 + (352*tmp60) + (123904*y3)), xmask & ymask)
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 + tmp65
    tl.store(out_ptr3 + (y0 + (20*x2) + (2478080*y1)), tmp66, xmask & ymask)
''')
