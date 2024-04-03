

# Original file: ./eca_botnext26ts_256___60.0/eca_botnext26ts_256___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/bfloat16/ko/ckoztlspqc5kjlpgzitbtjoiy3llqu54ooiqiu4zcqpjjmieh5ex.py
# Source Nodes: [l__mod___stem_conv3_bn_act, l__mod___stem_pool], Original ATen: [aten.max_pool2d_with_indices, aten.silu]
# l__mod___stem_conv3_bn_act => convert_element_type_14, mul_11, sigmoid_2
# l__mod___stem_pool => max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_silu_4 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_silu_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_silu_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_silu_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4096) % 64
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x5 = (xindex // 4096)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-8256) + x0 + (128*x1) + (16384*x5)), tmp10, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.where(tmp10, tmp14, float("-inf"))
    tmp16 = 2*x1
    tmp17 = tmp16 >= tmp1
    tmp18 = tmp16 < tmp3
    tmp19 = tmp17 & tmp18
    tmp20 = tmp5 & tmp19
    tmp21 = tl.load(in_ptr0 + ((-8192) + x0 + (128*x1) + (16384*x5)), tmp20, other=0.0)
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.where(tmp20, tmp24, float("-inf"))
    tmp26 = triton_helpers.maximum(tmp25, tmp15)
    tmp27 = 1 + (2*x1)
    tmp28 = tmp27 >= tmp1
    tmp29 = tmp27 < tmp3
    tmp30 = tmp28 & tmp29
    tmp31 = tmp5 & tmp30
    tmp32 = tl.load(in_ptr0 + ((-8128) + x0 + (128*x1) + (16384*x5)), tmp31, other=0.0)
    tmp33 = tl.sigmoid(tmp32)
    tmp34 = tmp32 * tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tl.where(tmp31, tmp35, float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp26)
    tmp38 = 2*x2
    tmp39 = tmp38 >= tmp1
    tmp40 = tmp38 < tmp3
    tmp41 = tmp39 & tmp40
    tmp42 = tmp41 & tmp9
    tmp43 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (16384*x5)), tmp42, other=0.0)
    tmp44 = tl.sigmoid(tmp43)
    tmp45 = tmp43 * tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tl.where(tmp42, tmp46, float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp37)
    tmp49 = tmp41 & tmp19
    tmp50 = tl.load(in_ptr0 + (x0 + (128*x1) + (16384*x5)), tmp49, other=0.0)
    tmp51 = tl.sigmoid(tmp50)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp52.to(tl.float32)
    tmp54 = tl.where(tmp49, tmp53, float("-inf"))
    tmp55 = triton_helpers.maximum(tmp54, tmp48)
    tmp56 = tmp41 & tmp30
    tmp57 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (16384*x5)), tmp56, other=0.0)
    tmp58 = tl.sigmoid(tmp57)
    tmp59 = tmp57 * tmp58
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tl.where(tmp56, tmp60, float("-inf"))
    tmp62 = triton_helpers.maximum(tmp61, tmp55)
    tmp63 = 1 + (2*x2)
    tmp64 = tmp63 >= tmp1
    tmp65 = tmp63 < tmp3
    tmp66 = tmp64 & tmp65
    tmp67 = tmp66 & tmp9
    tmp68 = tl.load(in_ptr0 + (8128 + x0 + (128*x1) + (16384*x5)), tmp67, other=0.0)
    tmp69 = tl.sigmoid(tmp68)
    tmp70 = tmp68 * tmp69
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tl.where(tmp67, tmp71, float("-inf"))
    tmp73 = triton_helpers.maximum(tmp72, tmp62)
    tmp74 = tmp66 & tmp19
    tmp75 = tl.load(in_ptr0 + (8192 + x0 + (128*x1) + (16384*x5)), tmp74, other=0.0)
    tmp76 = tl.sigmoid(tmp75)
    tmp77 = tmp75 * tmp76
    tmp78 = tmp77.to(tl.float32)
    tmp79 = tl.where(tmp74, tmp78, float("-inf"))
    tmp80 = triton_helpers.maximum(tmp79, tmp73)
    tmp81 = tmp66 & tmp30
    tmp82 = tl.load(in_ptr0 + (8256 + x0 + (128*x1) + (16384*x5)), tmp81, other=0.0)
    tmp83 = tl.sigmoid(tmp82)
    tmp84 = tmp82 * tmp83
    tmp85 = tmp84.to(tl.float32)
    tmp86 = tl.where(tmp81, tmp85, float("-inf"))
    tmp87 = triton_helpers.maximum(tmp86, tmp80)
    tl.store(out_ptr0 + (x6), tmp87, None)
''')
