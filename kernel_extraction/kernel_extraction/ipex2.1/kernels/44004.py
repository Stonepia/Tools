

# Original file: ./yolov3___60.0/yolov3___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/av/cavxqjhze3hizrhlou5i5tx2tutfet6qsxeieco4uyibcg6xtf47.py
# Source Nodes: [cat_7, l__mod___module_list_77_activation, l__mod___module_list_78], Original ATen: [aten.cat, aten.leaky_relu, aten.max_pool2d_with_indices]
# cat_7 => cat
# l__mod___module_list_77_activation => convert_element_type_274, gt_54, mul_219, where_54
# l__mod___module_list_78 => max_pool2d_with_indices
triton_poi_fused_cat_leaky_relu_max_pool2d_with_indices_20 = async_compile.triton('triton_poi_fused_cat_leaky_relu_max_pool2d_with_indices_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_leaky_relu_max_pool2d_with_indices_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_cat_leaky_relu_max_pool2d_with_indices_20(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 16)
    x2 = xindex % 16
    x5 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    y4 = yindex
    tmp244 = tl.load(in_ptr0 + (y0 + (512*x5) + (98304*y1)), xmask, eviction_policy='evict_last')
    tmp0 = (-2) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 12, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1, 1], 16, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-17408) + y0 + (512*x5) + (98304*y1)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = 0.0
    tmp14 = tmp12 > tmp13
    tmp15 = 0.1
    tmp16 = tmp12 * tmp15
    tmp17 = tl.where(tmp14, tmp12, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.where(tmp11, tmp18, float("-inf"))
    tmp20 = (-1) + x2
    tmp21 = tmp20 >= tmp1
    tmp22 = tmp20 < tmp8
    tmp23 = tmp21 & tmp22
    tmp24 = tmp5 & tmp23
    tmp25 = tl.load(in_ptr0 + ((-16896) + y0 + (512*x5) + (98304*y1)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp25 > tmp13
    tmp27 = tmp25 * tmp15
    tmp28 = tl.where(tmp26, tmp25, tmp27)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.where(tmp24, tmp29, float("-inf"))
    tmp31 = triton_helpers.maximum(tmp30, tmp19)
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp8
    tmp35 = tmp33 & tmp34
    tmp36 = tmp5 & tmp35
    tmp37 = tl.load(in_ptr0 + ((-16384) + y0 + (512*x5) + (98304*y1)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp37 > tmp13
    tmp39 = tmp37 * tmp15
    tmp40 = tl.where(tmp38, tmp37, tmp39)
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tl.where(tmp36, tmp41, float("-inf"))
    tmp43 = triton_helpers.maximum(tmp42, tmp31)
    tmp44 = 1 + x2
    tmp45 = tmp44 >= tmp1
    tmp46 = tmp44 < tmp8
    tmp47 = tmp45 & tmp46
    tmp48 = tmp5 & tmp47
    tmp49 = tl.load(in_ptr0 + ((-15872) + y0 + (512*x5) + (98304*y1)), tmp48 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tmp49 > tmp13
    tmp51 = tmp49 * tmp15
    tmp52 = tl.where(tmp50, tmp49, tmp51)
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tl.where(tmp48, tmp53, float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp43)
    tmp56 = 2 + x2
    tmp57 = tmp56 >= tmp1
    tmp58 = tmp56 < tmp8
    tmp59 = tmp57 & tmp58
    tmp60 = tmp5 & tmp59
    tmp61 = tl.load(in_ptr0 + ((-15360) + y0 + (512*x5) + (98304*y1)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tmp61 > tmp13
    tmp63 = tmp61 * tmp15
    tmp64 = tl.where(tmp62, tmp61, tmp63)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tl.where(tmp60, tmp65, float("-inf"))
    tmp67 = triton_helpers.maximum(tmp66, tmp55)
    tmp68 = (-1) + x3
    tmp69 = tmp68 >= tmp1
    tmp70 = tmp68 < tmp3
    tmp71 = tmp69 & tmp70
    tmp72 = tmp71 & tmp10
    tmp73 = tl.load(in_ptr0 + ((-9216) + y0 + (512*x5) + (98304*y1)), tmp72 & xmask, eviction_policy='evict_last', other=0.0)
    tmp74 = tmp73 > tmp13
    tmp75 = tmp73 * tmp15
    tmp76 = tl.where(tmp74, tmp73, tmp75)
    tmp77 = tmp76.to(tl.float32)
    tmp78 = tl.where(tmp72, tmp77, float("-inf"))
    tmp79 = triton_helpers.maximum(tmp78, tmp67)
    tmp80 = tmp71 & tmp23
    tmp81 = tl.load(in_ptr0 + ((-8704) + y0 + (512*x5) + (98304*y1)), tmp80 & xmask, eviction_policy='evict_last', other=0.0)
    tmp82 = tmp81 > tmp13
    tmp83 = tmp81 * tmp15
    tmp84 = tl.where(tmp82, tmp81, tmp83)
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tl.where(tmp80, tmp85, float("-inf"))
    tmp87 = triton_helpers.maximum(tmp86, tmp79)
    tmp88 = tmp71 & tmp35
    tmp89 = tl.load(in_ptr0 + ((-8192) + y0 + (512*x5) + (98304*y1)), tmp88 & xmask, eviction_policy='evict_last', other=0.0)
    tmp90 = tmp89 > tmp13
    tmp91 = tmp89 * tmp15
    tmp92 = tl.where(tmp90, tmp89, tmp91)
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tl.where(tmp88, tmp93, float("-inf"))
    tmp95 = triton_helpers.maximum(tmp94, tmp87)
    tmp96 = tmp71 & tmp47
    tmp97 = tl.load(in_ptr0 + ((-7680) + y0 + (512*x5) + (98304*y1)), tmp96 & xmask, eviction_policy='evict_last', other=0.0)
    tmp98 = tmp97 > tmp13
    tmp99 = tmp97 * tmp15
    tmp100 = tl.where(tmp98, tmp97, tmp99)
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tl.where(tmp96, tmp101, float("-inf"))
    tmp103 = triton_helpers.maximum(tmp102, tmp95)
    tmp104 = tmp71 & tmp59
    tmp105 = tl.load(in_ptr0 + ((-7168) + y0 + (512*x5) + (98304*y1)), tmp104 & xmask, eviction_policy='evict_last', other=0.0)
    tmp106 = tmp105 > tmp13
    tmp107 = tmp105 * tmp15
    tmp108 = tl.where(tmp106, tmp105, tmp107)
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tl.where(tmp104, tmp109, float("-inf"))
    tmp111 = triton_helpers.maximum(tmp110, tmp103)
    tmp112 = x3
    tmp113 = tmp112 >= tmp1
    tmp114 = tmp112 < tmp3
    tmp115 = tmp113 & tmp114
    tmp116 = tmp115 & tmp10
    tmp117 = tl.load(in_ptr0 + ((-1024) + y0 + (512*x5) + (98304*y1)), tmp116 & xmask, eviction_policy='evict_last', other=0.0)
    tmp118 = tmp117 > tmp13
    tmp119 = tmp117 * tmp15
    tmp120 = tl.where(tmp118, tmp117, tmp119)
    tmp121 = tmp120.to(tl.float32)
    tmp122 = tl.where(tmp116, tmp121, float("-inf"))
    tmp123 = triton_helpers.maximum(tmp122, tmp111)
    tmp124 = tmp115 & tmp23
    tmp125 = tl.load(in_ptr0 + ((-512) + y0 + (512*x5) + (98304*y1)), tmp124 & xmask, eviction_policy='evict_last', other=0.0)
    tmp126 = tmp125 > tmp13
    tmp127 = tmp125 * tmp15
    tmp128 = tl.where(tmp126, tmp125, tmp127)
    tmp129 = tmp128.to(tl.float32)
    tmp130 = tl.where(tmp124, tmp129, float("-inf"))
    tmp131 = triton_helpers.maximum(tmp130, tmp123)
    tmp132 = tmp115 & tmp35
    tmp133 = tl.load(in_ptr0 + (y0 + (512*x5) + (98304*y1)), tmp132 & xmask, eviction_policy='evict_last', other=0.0)
    tmp134 = tmp133 > tmp13
    tmp135 = tmp133 * tmp15
    tmp136 = tl.where(tmp134, tmp133, tmp135)
    tmp137 = tmp136.to(tl.float32)
    tmp138 = tl.where(tmp132, tmp137, float("-inf"))
    tmp139 = triton_helpers.maximum(tmp138, tmp131)
    tmp140 = tmp115 & tmp47
    tmp141 = tl.load(in_ptr0 + (512 + y0 + (512*x5) + (98304*y1)), tmp140 & xmask, eviction_policy='evict_last', other=0.0)
    tmp142 = tmp141 > tmp13
    tmp143 = tmp141 * tmp15
    tmp144 = tl.where(tmp142, tmp141, tmp143)
    tmp145 = tmp144.to(tl.float32)
    tmp146 = tl.where(tmp140, tmp145, float("-inf"))
    tmp147 = triton_helpers.maximum(tmp146, tmp139)
    tmp148 = tmp115 & tmp59
    tmp149 = tl.load(in_ptr0 + (1024 + y0 + (512*x5) + (98304*y1)), tmp148 & xmask, eviction_policy='evict_last', other=0.0)
    tmp150 = tmp149 > tmp13
    tmp151 = tmp149 * tmp15
    tmp152 = tl.where(tmp150, tmp149, tmp151)
    tmp153 = tmp152.to(tl.float32)
    tmp154 = tl.where(tmp148, tmp153, float("-inf"))
    tmp155 = triton_helpers.maximum(tmp154, tmp147)
    tmp156 = 1 + x3
    tmp157 = tmp156 >= tmp1
    tmp158 = tmp156 < tmp3
    tmp159 = tmp157 & tmp158
    tmp160 = tmp159 & tmp10
    tmp161 = tl.load(in_ptr0 + (7168 + y0 + (512*x5) + (98304*y1)), tmp160 & xmask, eviction_policy='evict_last', other=0.0)
    tmp162 = tmp161 > tmp13
    tmp163 = tmp161 * tmp15
    tmp164 = tl.where(tmp162, tmp161, tmp163)
    tmp165 = tmp164.to(tl.float32)
    tmp166 = tl.where(tmp160, tmp165, float("-inf"))
    tmp167 = triton_helpers.maximum(tmp166, tmp155)
    tmp168 = tmp159 & tmp23
    tmp169 = tl.load(in_ptr0 + (7680 + y0 + (512*x5) + (98304*y1)), tmp168 & xmask, eviction_policy='evict_last', other=0.0)
    tmp170 = tmp169 > tmp13
    tmp171 = tmp169 * tmp15
    tmp172 = tl.where(tmp170, tmp169, tmp171)
    tmp173 = tmp172.to(tl.float32)
    tmp174 = tl.where(tmp168, tmp173, float("-inf"))
    tmp175 = triton_helpers.maximum(tmp174, tmp167)
    tmp176 = tmp159 & tmp35
    tmp177 = tl.load(in_ptr0 + (8192 + y0 + (512*x5) + (98304*y1)), tmp176 & xmask, eviction_policy='evict_last', other=0.0)
    tmp178 = tmp177 > tmp13
    tmp179 = tmp177 * tmp15
    tmp180 = tl.where(tmp178, tmp177, tmp179)
    tmp181 = tmp180.to(tl.float32)
    tmp182 = tl.where(tmp176, tmp181, float("-inf"))
    tmp183 = triton_helpers.maximum(tmp182, tmp175)
    tmp184 = tmp159 & tmp47
    tmp185 = tl.load(in_ptr0 + (8704 + y0 + (512*x5) + (98304*y1)), tmp184 & xmask, eviction_policy='evict_last', other=0.0)
    tmp186 = tmp185 > tmp13
    tmp187 = tmp185 * tmp15
    tmp188 = tl.where(tmp186, tmp185, tmp187)
    tmp189 = tmp188.to(tl.float32)
    tmp190 = tl.where(tmp184, tmp189, float("-inf"))
    tmp191 = triton_helpers.maximum(tmp190, tmp183)
    tmp192 = tmp159 & tmp59
    tmp193 = tl.load(in_ptr0 + (9216 + y0 + (512*x5) + (98304*y1)), tmp192 & xmask, eviction_policy='evict_last', other=0.0)
    tmp194 = tmp193 > tmp13
    tmp195 = tmp193 * tmp15
    tmp196 = tl.where(tmp194, tmp193, tmp195)
    tmp197 = tmp196.to(tl.float32)
    tmp198 = tl.where(tmp192, tmp197, float("-inf"))
    tmp199 = triton_helpers.maximum(tmp198, tmp191)
    tmp200 = 2 + x3
    tmp201 = tmp200 >= tmp1
    tmp202 = tmp200 < tmp3
    tmp203 = tmp201 & tmp202
    tmp204 = tmp203 & tmp10
    tmp205 = tl.load(in_ptr0 + (15360 + y0 + (512*x5) + (98304*y1)), tmp204 & xmask, eviction_policy='evict_last', other=0.0)
    tmp206 = tmp205 > tmp13
    tmp207 = tmp205 * tmp15
    tmp208 = tl.where(tmp206, tmp205, tmp207)
    tmp209 = tmp208.to(tl.float32)
    tmp210 = tl.where(tmp204, tmp209, float("-inf"))
    tmp211 = triton_helpers.maximum(tmp210, tmp199)
    tmp212 = tmp203 & tmp23
    tmp213 = tl.load(in_ptr0 + (15872 + y0 + (512*x5) + (98304*y1)), tmp212 & xmask, eviction_policy='evict_last', other=0.0)
    tmp214 = tmp213 > tmp13
    tmp215 = tmp213 * tmp15
    tmp216 = tl.where(tmp214, tmp213, tmp215)
    tmp217 = tmp216.to(tl.float32)
    tmp218 = tl.where(tmp212, tmp217, float("-inf"))
    tmp219 = triton_helpers.maximum(tmp218, tmp211)
    tmp220 = tmp203 & tmp35
    tmp221 = tl.load(in_ptr0 + (16384 + y0 + (512*x5) + (98304*y1)), tmp220 & xmask, eviction_policy='evict_last', other=0.0)
    tmp222 = tmp221 > tmp13
    tmp223 = tmp221 * tmp15
    tmp224 = tl.where(tmp222, tmp221, tmp223)
    tmp225 = tmp224.to(tl.float32)
    tmp226 = tl.where(tmp220, tmp225, float("-inf"))
    tmp227 = triton_helpers.maximum(tmp226, tmp219)
    tmp228 = tmp203 & tmp47
    tmp229 = tl.load(in_ptr0 + (16896 + y0 + (512*x5) + (98304*y1)), tmp228 & xmask, eviction_policy='evict_last', other=0.0)
    tmp230 = tmp229 > tmp13
    tmp231 = tmp229 * tmp15
    tmp232 = tl.where(tmp230, tmp229, tmp231)
    tmp233 = tmp232.to(tl.float32)
    tmp234 = tl.where(tmp228, tmp233, float("-inf"))
    tmp235 = triton_helpers.maximum(tmp234, tmp227)
    tmp236 = tmp203 & tmp59
    tmp237 = tl.load(in_ptr0 + (17408 + y0 + (512*x5) + (98304*y1)), tmp236 & xmask, eviction_policy='evict_last', other=0.0)
    tmp238 = tmp237 > tmp13
    tmp239 = tmp237 * tmp15
    tmp240 = tl.where(tmp238, tmp237, tmp239)
    tmp241 = tmp240.to(tl.float32)
    tmp242 = tl.where(tmp236, tmp241, float("-inf"))
    tmp243 = triton_helpers.maximum(tmp242, tmp235)
    tmp245 = tmp244 > tmp13
    tmp246 = tmp244 * tmp15
    tmp247 = tl.where(tmp245, tmp244, tmp246)
    tmp248 = tmp247.to(tl.float32)
    tl.store(out_ptr0 + (x5 + (192*y0) + (393216*y1)), tmp243, xmask)
    tl.store(out_ptr1 + (x5 + (192*y4)), tmp248, xmask)
    tl.store(out_ptr2 + (x5 + (192*y0) + (393216*y1)), tmp248, xmask)
''')
