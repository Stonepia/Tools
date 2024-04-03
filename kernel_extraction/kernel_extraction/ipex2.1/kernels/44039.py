

# Original file: ./yolov3___60.0/yolov3___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/m7/cm7whtadbbfrf5mxhgqvthrmjapstgmwnbi2tos2yvvbicrmboqk.py
# Source Nodes: [cat_7, l__self___module_list_78], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# cat_7 => cat
# l__self___module_list_78 => max_pool2d_with_indices
triton_poi_fused_cat_max_pool2d_with_indices_19 = async_compile.triton('triton_poi_fused_cat_max_pool2d_with_indices_19', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_max_pool2d_with_indices_19(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8192) % 12
    x1 = (xindex // 512) % 16
    x6 = xindex
    x0 = xindex % 512
    x7 = (xindex // 512)
    tmp142 = tl.load(in_ptr0 + (x6), None).to(tl.float32)
    tmp0 = (-2) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 16, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-17408) + x6), tmp11, other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp11, tmp12, float("-inf"))
    tmp14 = (-1) + x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp8
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-16896) + x6), tmp18, other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp18, tmp19, float("-inf"))
    tmp21 = triton_helpers.maximum(tmp20, tmp13)
    tmp22 = x1
    tmp23 = tmp22 >= tmp1
    tmp24 = tmp22 < tmp8
    tmp25 = tmp23 & tmp24
    tmp26 = tmp5 & tmp25
    tmp27 = tl.load(in_ptr0 + ((-16384) + x6), tmp26, other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp26, tmp27, float("-inf"))
    tmp29 = triton_helpers.maximum(tmp28, tmp21)
    tmp30 = 1 + x1
    tmp31 = tmp30 >= tmp1
    tmp32 = tmp30 < tmp8
    tmp33 = tmp31 & tmp32
    tmp34 = tmp5 & tmp33
    tmp35 = tl.load(in_ptr0 + ((-15872) + x6), tmp34, other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp34, tmp35, float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp29)
    tmp38 = 2 + x1
    tmp39 = tmp38 >= tmp1
    tmp40 = tmp38 < tmp8
    tmp41 = tmp39 & tmp40
    tmp42 = tmp5 & tmp41
    tmp43 = tl.load(in_ptr0 + ((-15360) + x6), tmp42, other=0.0).to(tl.float32)
    tmp44 = tl.where(tmp42, tmp43, float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp37)
    tmp46 = (-1) + x2
    tmp47 = tmp46 >= tmp1
    tmp48 = tmp46 < tmp3
    tmp49 = tmp47 & tmp48
    tmp50 = tmp49 & tmp10
    tmp51 = tl.load(in_ptr0 + ((-9216) + x6), tmp50, other=0.0).to(tl.float32)
    tmp52 = tl.where(tmp50, tmp51, float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp45)
    tmp54 = tmp49 & tmp17
    tmp55 = tl.load(in_ptr0 + ((-8704) + x6), tmp54, other=0.0).to(tl.float32)
    tmp56 = tl.where(tmp54, tmp55, float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tmp58 = tmp49 & tmp25
    tmp59 = tl.load(in_ptr0 + ((-8192) + x6), tmp58, other=0.0).to(tl.float32)
    tmp60 = tl.where(tmp58, tmp59, float("-inf"))
    tmp61 = triton_helpers.maximum(tmp60, tmp57)
    tmp62 = tmp49 & tmp33
    tmp63 = tl.load(in_ptr0 + ((-7680) + x6), tmp62, other=0.0).to(tl.float32)
    tmp64 = tl.where(tmp62, tmp63, float("-inf"))
    tmp65 = triton_helpers.maximum(tmp64, tmp61)
    tmp66 = tmp49 & tmp41
    tmp67 = tl.load(in_ptr0 + ((-7168) + x6), tmp66, other=0.0).to(tl.float32)
    tmp68 = tl.where(tmp66, tmp67, float("-inf"))
    tmp69 = triton_helpers.maximum(tmp68, tmp65)
    tmp70 = x2
    tmp71 = tmp70 >= tmp1
    tmp72 = tmp70 < tmp3
    tmp73 = tmp71 & tmp72
    tmp74 = tmp73 & tmp10
    tmp75 = tl.load(in_ptr0 + ((-1024) + x6), tmp74, other=0.0).to(tl.float32)
    tmp76 = tl.where(tmp74, tmp75, float("-inf"))
    tmp77 = triton_helpers.maximum(tmp76, tmp69)
    tmp78 = tmp73 & tmp17
    tmp79 = tl.load(in_ptr0 + ((-512) + x6), tmp78, other=0.0).to(tl.float32)
    tmp80 = tl.where(tmp78, tmp79, float("-inf"))
    tmp81 = triton_helpers.maximum(tmp80, tmp77)
    tmp82 = tmp73 & tmp25
    tmp83 = tl.load(in_ptr0 + (x6), tmp82, other=0.0).to(tl.float32)
    tmp84 = tl.where(tmp82, tmp83, float("-inf"))
    tmp85 = triton_helpers.maximum(tmp84, tmp81)
    tmp86 = tmp73 & tmp33
    tmp87 = tl.load(in_ptr0 + (512 + x6), tmp86, other=0.0).to(tl.float32)
    tmp88 = tl.where(tmp86, tmp87, float("-inf"))
    tmp89 = triton_helpers.maximum(tmp88, tmp85)
    tmp90 = tmp73 & tmp41
    tmp91 = tl.load(in_ptr0 + (1024 + x6), tmp90, other=0.0).to(tl.float32)
    tmp92 = tl.where(tmp90, tmp91, float("-inf"))
    tmp93 = triton_helpers.maximum(tmp92, tmp89)
    tmp94 = 1 + x2
    tmp95 = tmp94 >= tmp1
    tmp96 = tmp94 < tmp3
    tmp97 = tmp95 & tmp96
    tmp98 = tmp97 & tmp10
    tmp99 = tl.load(in_ptr0 + (7168 + x6), tmp98, other=0.0).to(tl.float32)
    tmp100 = tl.where(tmp98, tmp99, float("-inf"))
    tmp101 = triton_helpers.maximum(tmp100, tmp93)
    tmp102 = tmp97 & tmp17
    tmp103 = tl.load(in_ptr0 + (7680 + x6), tmp102, other=0.0).to(tl.float32)
    tmp104 = tl.where(tmp102, tmp103, float("-inf"))
    tmp105 = triton_helpers.maximum(tmp104, tmp101)
    tmp106 = tmp97 & tmp25
    tmp107 = tl.load(in_ptr0 + (8192 + x6), tmp106, other=0.0).to(tl.float32)
    tmp108 = tl.where(tmp106, tmp107, float("-inf"))
    tmp109 = triton_helpers.maximum(tmp108, tmp105)
    tmp110 = tmp97 & tmp33
    tmp111 = tl.load(in_ptr0 + (8704 + x6), tmp110, other=0.0).to(tl.float32)
    tmp112 = tl.where(tmp110, tmp111, float("-inf"))
    tmp113 = triton_helpers.maximum(tmp112, tmp109)
    tmp114 = tmp97 & tmp41
    tmp115 = tl.load(in_ptr0 + (9216 + x6), tmp114, other=0.0).to(tl.float32)
    tmp116 = tl.where(tmp114, tmp115, float("-inf"))
    tmp117 = triton_helpers.maximum(tmp116, tmp113)
    tmp118 = 2 + x2
    tmp119 = tmp118 >= tmp1
    tmp120 = tmp118 < tmp3
    tmp121 = tmp119 & tmp120
    tmp122 = tmp121 & tmp10
    tmp123 = tl.load(in_ptr0 + (15360 + x6), tmp122, other=0.0).to(tl.float32)
    tmp124 = tl.where(tmp122, tmp123, float("-inf"))
    tmp125 = triton_helpers.maximum(tmp124, tmp117)
    tmp126 = tmp121 & tmp17
    tmp127 = tl.load(in_ptr0 + (15872 + x6), tmp126, other=0.0).to(tl.float32)
    tmp128 = tl.where(tmp126, tmp127, float("-inf"))
    tmp129 = triton_helpers.maximum(tmp128, tmp125)
    tmp130 = tmp121 & tmp25
    tmp131 = tl.load(in_ptr0 + (16384 + x6), tmp130, other=0.0).to(tl.float32)
    tmp132 = tl.where(tmp130, tmp131, float("-inf"))
    tmp133 = triton_helpers.maximum(tmp132, tmp129)
    tmp134 = tmp121 & tmp33
    tmp135 = tl.load(in_ptr0 + (16896 + x6), tmp134, other=0.0).to(tl.float32)
    tmp136 = tl.where(tmp134, tmp135, float("-inf"))
    tmp137 = triton_helpers.maximum(tmp136, tmp133)
    tmp138 = tmp121 & tmp41
    tmp139 = tl.load(in_ptr0 + (17408 + x6), tmp138, other=0.0).to(tl.float32)
    tmp140 = tl.where(tmp138, tmp139, float("-inf"))
    tmp141 = triton_helpers.maximum(tmp140, tmp137)
    tl.store(out_ptr0 + (x0 + (2048*x7)), tmp141, None)
    tl.store(out_ptr1 + (x0 + (2048*x7)), tmp142, None)
''')
