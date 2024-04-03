

# Original file: ./crossvit_9_240___60.0/crossvit_9_240___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float16/p2/cp25otggmrm3riqemaqu5erlwooajrjp5p22gkkrti4j2fwwujr5.py
# Source Nodes: [interpolate], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.floor, aten.mul, aten.rsub, aten.sub]
# interpolate => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add_12, add_13, add_14, add_2, add_20, add_21, add_22, add_28, add_29, add_30, add_36, add_37, add_38, add_39, add_40, add_41, add_42, add_43, add_44, add_45, add_46, convert_element_type, convert_element_type_3, floor_1, mul_1, mul_14, mul_15, mul_16, mul_17, mul_30, mul_31, mul_32, mul_33, mul_46, mul_47, mul_48, mul_49, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_80, mul_81, sub_2, sub_3, sub_38, sub_39, sub_40, sub_41, sub_42, sub_43, sub_44, sub_45
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

@pointwise(size_hints=[512, 65536], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_floor_mul_rsub_sub_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]})
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
    tmp26 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp106 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp108 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp111 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp114 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
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
    tmp24 = tl.load(in_ptr0 + (tmp23 + (240*tmp12) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = triton_helpers.maximum(tmp19, tmp9)
    tmp29 = triton_helpers.minimum(tmp28, tmp11)
    tmp30 = tl.load(in_ptr0 + (tmp29 + (240*tmp12) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp33 = tmp31 * tmp32
    tmp34 = tmp27 + tmp33
    tmp35 = tmp19 + tmp20
    tmp36 = triton_helpers.maximum(tmp35, tmp9)
    tmp37 = triton_helpers.minimum(tmp36, tmp11)
    tmp38 = tl.load(in_ptr0 + (tmp37 + (240*tmp12) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp39 = tmp38.to(tl.float32)
    tmp41 = tmp39 * tmp40
    tmp42 = tmp34 + tmp41
    tmp43 = tl.full([1, 1], 2, tl.int64)
    tmp44 = tmp19 + tmp43
    tmp45 = triton_helpers.maximum(tmp44, tmp9)
    tmp46 = triton_helpers.minimum(tmp45, tmp11)
    tmp47 = tl.load(in_ptr0 + (tmp46 + (240*tmp12) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp48 = tmp47.to(tl.float32)
    tmp50 = tmp48 * tmp49
    tmp51 = tmp42 + tmp50
    tmp52 = tmp8 - tmp20
    tmp53 = triton_helpers.maximum(tmp52, tmp9)
    tmp54 = triton_helpers.minimum(tmp53, tmp11)
    tmp55 = tl.load(in_ptr0 + (tmp23 + (240*tmp54) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 * tmp26
    tmp58 = tl.load(in_ptr0 + (tmp29 + (240*tmp54) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp59 * tmp32
    tmp61 = tmp57 + tmp60
    tmp62 = tmp8 + tmp20
    tmp63 = triton_helpers.maximum(tmp62, tmp9)
    tmp64 = triton_helpers.minimum(tmp63, tmp11)
    tmp65 = tl.load(in_ptr0 + (tmp23 + (240*tmp64) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp66 = tmp65.to(tl.float32)
    tmp67 = tmp66 * tmp26
    tmp68 = tl.load(in_ptr0 + (tmp29 + (240*tmp64) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp69 = tmp68.to(tl.float32)
    tmp70 = tmp69 * tmp32
    tmp71 = tmp67 + tmp70
    tmp72 = tmp8 + tmp43
    tmp73 = triton_helpers.maximum(tmp72, tmp9)
    tmp74 = triton_helpers.minimum(tmp73, tmp11)
    tmp75 = tl.load(in_ptr0 + (tmp23 + (240*tmp74) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp76 = tmp75.to(tl.float32)
    tmp77 = tmp76 * tmp26
    tmp78 = tl.load(in_ptr0 + (tmp29 + (240*tmp74) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp79 = tmp78.to(tl.float32)
    tmp80 = tmp79 * tmp32
    tmp81 = tmp77 + tmp80
    tmp82 = tl.load(in_ptr0 + (tmp37 + (240*tmp54) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp83 = tmp82.to(tl.float32)
    tmp84 = tmp83 * tmp40
    tmp85 = tmp61 + tmp84
    tmp86 = tl.load(in_ptr0 + (tmp46 + (240*tmp54) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp87 = tmp86.to(tl.float32)
    tmp88 = tmp87 * tmp49
    tmp89 = tmp85 + tmp88
    tmp90 = tl.load(in_ptr0 + (tmp37 + (240*tmp64) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp91 = tmp90.to(tl.float32)
    tmp92 = tmp91 * tmp40
    tmp93 = tmp71 + tmp92
    tmp94 = tl.load(in_ptr0 + (tmp46 + (240*tmp64) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp95 = tmp94.to(tl.float32)
    tmp96 = tmp95 * tmp49
    tmp97 = tmp93 + tmp96
    tmp98 = tl.load(in_ptr0 + (tmp37 + (240*tmp74) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp99 = tmp98.to(tl.float32)
    tmp100 = tmp99 * tmp40
    tmp101 = tmp81 + tmp100
    tmp102 = tl.load(in_ptr0 + (tmp46 + (240*tmp74) + (57600*y0)), xmask & ymask).to(tl.float32)
    tmp103 = tmp102.to(tl.float32)
    tmp104 = tmp103 * tmp49
    tmp105 = tmp101 + tmp104
    tmp107 = tmp89 * tmp106
    tmp109 = tmp51 * tmp108
    tmp110 = tmp107 + tmp109
    tmp112 = tmp97 * tmp111
    tmp113 = tmp110 + tmp112
    tmp115 = tmp105 * tmp114
    tmp116 = tmp113 + tmp115
    tmp117 = tmp116.to(tl.float32)
    tl.store(out_ptr0 + (y4 + (3*x3) + (150528*y5)), tmp117, xmask & ymask)
''')
