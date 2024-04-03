

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/jh/cjhaew4ll7au74kbqxerhijzb2oagae5ucha6jbrqxeuahb7erxg.py
# Source Nodes: [interpolate, l__self___patch_embed_1_proj], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.floor, aten.mul, aten.rsub, aten.sub]
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add_12, add_13, add_14, add_2, add_20, add_21, add_22, add_28, add_29, add_30, add_36, add_37, add_38, add_39, add_40, add_41, add_42, add_43, add_44, add_45, add_46, floor_1, mul_1, mul_14, mul_15, mul_16, mul_17, mul_30, mul_31, mul_32, mul_33, mul_46, mul_47, mul_48, mul_49, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, sub_2, sub_3, sub_38, sub_39, sub_40, sub_41, sub_42, sub_43, sub_44, sub_45
# l__self___patch_embed_1_proj => convert_element_type_6
triton_poi_fused__to_copy__unsafe_index_add_floor_mul_rsub_sub_14 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_floor_mul_rsub_sub_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[512, 65536], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_floor_mul_rsub_sub_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_floor_mul_rsub_sub_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = (xindex // 224)
    x1 = xindex % 224
    y0 = yindex
    x3 = xindex
    y4 = yindex % 3
    y5 = (yindex // 3)
    tmp25 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp90 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp98 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.0714285714285714
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1, 1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1, 1], 239, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tmp13 = x1
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 + tmp2
    tmp16 = tmp15 * tmp4
    tmp17 = tmp16 - tmp2
    tmp18 = libdevice.floor(tmp17)
    tmp19 = tmp18.to(tl.int32)
    tmp20 = tl.full([1, 1], 1, tl.int64)
    tmp21 = tmp19 - tmp20
    tmp22 = triton_helpers.maximum(tmp21, tmp9)
    tmp23 = triton_helpers.minimum(tmp22, tmp11)
    tmp24 = tl.load(in_ptr0 + (tmp23 + (240*tmp12) + (57600*y0)), xmask & ymask)
    tmp26 = tmp24 * tmp25
    tmp27 = triton_helpers.maximum(tmp19, tmp9)
    tmp28 = triton_helpers.minimum(tmp27, tmp11)
    tmp29 = tl.load(in_ptr0 + (tmp28 + (240*tmp12) + (57600*y0)), xmask & ymask)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp26 + tmp31
    tmp33 = tmp19 + tmp20
    tmp34 = triton_helpers.maximum(tmp33, tmp9)
    tmp35 = triton_helpers.minimum(tmp34, tmp11)
    tmp36 = tl.load(in_ptr0 + (tmp35 + (240*tmp12) + (57600*y0)), xmask & ymask)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp32 + tmp38
    tmp40 = tl.full([1, 1], 2, tl.int64)
    tmp41 = tmp19 + tmp40
    tmp42 = triton_helpers.maximum(tmp41, tmp9)
    tmp43 = triton_helpers.minimum(tmp42, tmp11)
    tmp44 = tl.load(in_ptr0 + (tmp43 + (240*tmp12) + (57600*y0)), xmask & ymask)
    tmp46 = tmp44 * tmp45
    tmp47 = tmp39 + tmp46
    tmp48 = tmp8 - tmp20
    tmp49 = triton_helpers.maximum(tmp48, tmp9)
    tmp50 = triton_helpers.minimum(tmp49, tmp11)
    tmp51 = tl.load(in_ptr0 + (tmp23 + (240*tmp50) + (57600*y0)), xmask & ymask)
    tmp52 = tmp51 * tmp25
    tmp53 = tl.load(in_ptr0 + (tmp28 + (240*tmp50) + (57600*y0)), xmask & ymask)
    tmp54 = tmp53 * tmp30
    tmp55 = tmp52 + tmp54
    tmp56 = tmp8 + tmp20
    tmp57 = triton_helpers.maximum(tmp56, tmp9)
    tmp58 = triton_helpers.minimum(tmp57, tmp11)
    tmp59 = tl.load(in_ptr0 + (tmp23 + (240*tmp58) + (57600*y0)), xmask & ymask)
    tmp60 = tmp59 * tmp25
    tmp61 = tl.load(in_ptr0 + (tmp28 + (240*tmp58) + (57600*y0)), xmask & ymask)
    tmp62 = tmp61 * tmp30
    tmp63 = tmp60 + tmp62
    tmp64 = tmp8 + tmp40
    tmp65 = triton_helpers.maximum(tmp64, tmp9)
    tmp66 = triton_helpers.minimum(tmp65, tmp11)
    tmp67 = tl.load(in_ptr0 + (tmp23 + (240*tmp66) + (57600*y0)), xmask & ymask)
    tmp68 = tmp67 * tmp25
    tmp69 = tl.load(in_ptr0 + (tmp28 + (240*tmp66) + (57600*y0)), xmask & ymask)
    tmp70 = tmp69 * tmp30
    tmp71 = tmp68 + tmp70
    tmp72 = tl.load(in_ptr0 + (tmp35 + (240*tmp50) + (57600*y0)), xmask & ymask)
    tmp73 = tmp72 * tmp37
    tmp74 = tmp55 + tmp73
    tmp75 = tl.load(in_ptr0 + (tmp43 + (240*tmp50) + (57600*y0)), xmask & ymask)
    tmp76 = tmp75 * tmp45
    tmp77 = tmp74 + tmp76
    tmp78 = tl.load(in_ptr0 + (tmp35 + (240*tmp58) + (57600*y0)), xmask & ymask)
    tmp79 = tmp78 * tmp37
    tmp80 = tmp63 + tmp79
    tmp81 = tl.load(in_ptr0 + (tmp43 + (240*tmp58) + (57600*y0)), xmask & ymask)
    tmp82 = tmp81 * tmp45
    tmp83 = tmp80 + tmp82
    tmp84 = tl.load(in_ptr0 + (tmp35 + (240*tmp66) + (57600*y0)), xmask & ymask)
    tmp85 = tmp84 * tmp37
    tmp86 = tmp71 + tmp85
    tmp87 = tl.load(in_ptr0 + (tmp43 + (240*tmp66) + (57600*y0)), xmask & ymask)
    tmp88 = tmp87 * tmp45
    tmp89 = tmp86 + tmp88
    tmp91 = tmp77 * tmp90
    tmp93 = tmp47 * tmp92
    tmp94 = tmp91 + tmp93
    tmp96 = tmp83 * tmp95
    tmp97 = tmp94 + tmp96
    tmp99 = tmp89 * tmp98
    tmp100 = tmp97 + tmp99
    tmp101 = tmp100.to(tl.float32)
    tl.store(out_ptr0 + (y4 + (3*x3) + (150528*y5)), tmp101, xmask & ymask)
''')
