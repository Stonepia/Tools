

# Original file: ./inception_v3___60.0/inception_v3___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ws/cwsphfg3ofojer3qczkg7ybimc32rvijyqlukbnx7lwcdhef5ko6.py
# Source Nodes: [avg_pool2d], Original ATen: [aten.avg_pool2d]
# avg_pool2d => avg_pool2d
triton_poi_fused_avg_pool2d_3 = async_compile.triton('triton_poi_fused_avg_pool2d_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[32768, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_avg_pool2d_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 35)
    x2 = xindex % 35
    x5 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6912) + y0 + (192*x5) + (235200*y1)), tmp10 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, 0.0)
    tmp13 = x2
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-6720) + y0 + (192*x5) + (235200*y1)), tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, 0.0)
    tmp20 = tmp19 + tmp12
    tmp21 = 1 + x2
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-6528) + y0 + (192*x5) + (235200*y1)), tmp25 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tmp27 + tmp20
    tmp29 = x3
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-192) + y0 + (192*x5) + (235200*y1)), tmp33 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, 0.0)
    tmp36 = tmp35 + tmp28
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp37 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, 0.0)
    tmp40 = tmp39 + tmp36
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (192 + y0 + (192*x5) + (235200*y1)), tmp41 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, 0.0)
    tmp44 = tmp43 + tmp40
    tmp45 = 1 + x3
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (6528 + y0 + (192*x5) + (235200*y1)), tmp49 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, 0.0)
    tmp52 = tmp51 + tmp44
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (6720 + y0 + (192*x5) + (235200*y1)), tmp53 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, 0.0)
    tmp56 = tmp55 + tmp52
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (6912 + y0 + (192*x5) + (235200*y1)), tmp57 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, 0.0)
    tmp60 = tmp59 + tmp56
    tmp61 = tl.full([1, 1], -1, tl.int64)
    tmp62 = tmp0 >= tmp61
    tmp63 = tl.full([1, 1], 36, tl.int64)
    tmp64 = tmp0 < tmp63
    tmp65 = tmp62 & tmp64
    tmp66 = tmp6 >= tmp61
    tmp67 = tmp6 < tmp63
    tmp68 = tmp66 & tmp67
    tmp69 = tmp65 & tmp68
    tmp70 = tl.broadcast_to((-1) + x3, [XBLOCK, YBLOCK])
    tmp71 = tmp70 >= tmp1
    tmp72 = tmp70 < tmp3
    tmp73 = tmp71 & tmp72
    tmp74 = tl.broadcast_to((-1) + x2, [XBLOCK, YBLOCK])
    tmp75 = tmp74 >= tmp1
    tmp76 = tmp74 < tmp3
    tmp77 = tmp75 & tmp76
    tmp78 = tmp73 & tmp77
    tmp79 = tmp78 & tmp69
    tmp80 = 1.0
    tmp81 = tl.where(tmp79, tmp80, 1.0)
    tmp82 = tl.where(tmp69, tmp81, 0.0)
    tmp83 = tmp13 >= tmp61
    tmp84 = tmp13 < tmp63
    tmp85 = tmp83 & tmp84
    tmp86 = tmp65 & tmp85
    tmp87 = tl.broadcast_to(x2, [XBLOCK, YBLOCK])
    tmp88 = tmp87 >= tmp1
    tmp89 = tmp87 < tmp3
    tmp90 = tmp88 & tmp89
    tmp91 = tmp73 & tmp90
    tmp92 = tmp91 & tmp86
    tmp93 = tl.where(tmp92, tmp80, 1.0)
    tmp94 = tl.where(tmp86, tmp93, 0.0)
    tmp95 = tmp94 + tmp82
    tmp96 = tmp21 >= tmp61
    tmp97 = tmp21 < tmp63
    tmp98 = tmp96 & tmp97
    tmp99 = tmp65 & tmp98
    tmp100 = tl.broadcast_to(1 + x2, [XBLOCK, YBLOCK])
    tmp101 = tmp100 >= tmp1
    tmp102 = tmp100 < tmp3
    tmp103 = tmp101 & tmp102
    tmp104 = tmp73 & tmp103
    tmp105 = tmp104 & tmp99
    tmp106 = tl.where(tmp105, tmp80, 1.0)
    tmp107 = tl.where(tmp99, tmp106, 0.0)
    tmp108 = tmp107 + tmp95
    tmp109 = tmp29 >= tmp61
    tmp110 = tmp29 < tmp63
    tmp111 = tmp109 & tmp110
    tmp112 = tmp111 & tmp68
    tmp113 = tl.broadcast_to(x3, [XBLOCK, YBLOCK])
    tmp114 = tmp113 >= tmp1
    tmp115 = tmp113 < tmp3
    tmp116 = tmp114 & tmp115
    tmp117 = tmp116 & tmp77
    tmp118 = tmp117 & tmp112
    tmp119 = tl.where(tmp118, tmp80, 1.0)
    tmp120 = tl.where(tmp112, tmp119, 0.0)
    tmp121 = tmp120 + tmp108
    tmp122 = tmp111 & tmp85
    tmp123 = tmp116 & tmp90
    tmp124 = tmp123 & tmp122
    tmp125 = tl.where(tmp124, tmp80, 1.0)
    tmp126 = tl.where(tmp122, tmp125, 0.0)
    tmp127 = tmp126 + tmp121
    tmp128 = tmp111 & tmp98
    tmp129 = tmp116 & tmp103
    tmp130 = tmp129 & tmp128
    tmp131 = tl.where(tmp130, tmp80, 1.0)
    tmp132 = tl.where(tmp128, tmp131, 0.0)
    tmp133 = tmp132 + tmp127
    tmp134 = tmp45 >= tmp61
    tmp135 = tmp45 < tmp63
    tmp136 = tmp134 & tmp135
    tmp137 = tmp136 & tmp68
    tmp138 = tl.broadcast_to(1 + x3, [XBLOCK, YBLOCK])
    tmp139 = tmp138 >= tmp1
    tmp140 = tmp138 < tmp3
    tmp141 = tmp139 & tmp140
    tmp142 = tmp141 & tmp77
    tmp143 = tmp142 & tmp137
    tmp144 = tl.where(tmp143, tmp80, 1.0)
    tmp145 = tl.where(tmp137, tmp144, 0.0)
    tmp146 = tmp145 + tmp133
    tmp147 = tmp136 & tmp85
    tmp148 = tmp141 & tmp90
    tmp149 = tmp148 & tmp147
    tmp150 = tl.where(tmp149, tmp80, 1.0)
    tmp151 = tl.where(tmp147, tmp150, 0.0)
    tmp152 = tmp151 + tmp146
    tmp153 = tmp136 & tmp98
    tmp154 = tmp141 & tmp103
    tmp155 = tmp154 & tmp153
    tmp156 = tl.where(tmp155, tmp80, 1.0)
    tmp157 = tl.where(tmp153, tmp156, 0.0)
    tmp158 = tmp157 + tmp152
    tmp159 = tmp60 / tmp158
    tl.store(out_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp159, xmask)
''')
