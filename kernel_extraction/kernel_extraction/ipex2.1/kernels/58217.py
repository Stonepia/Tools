

# Original file: ./vision_maskrcnn__24_inference_64.4/vision_maskrcnn__24_inference_64.4.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/vj/cvj7zcs2p22zpkwveva4q6idtddignt23vqqn2rpssehegkhv2qr.py
# Source Nodes: [add_1, l__self___backbone_body_maxpool, l__self___backbone_body_relu, mul_2], Original ATen: [aten.add, aten.max_pool2d_with_indices, aten.mul, aten.relu]
# add_1 => add_1
# l__self___backbone_body_maxpool => max_pool2d_with_indices
# l__self___backbone_body_relu => relu
# mul_2 => mul_2
triton_poi_fused_add_max_pool2d_with_indices_mul_relu_2 = async_compile.triton('triton_poi_fused_add_max_pool2d_with_indices_mul_relu_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_max_pool2d_with_indices_mul_relu_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_max_pool2d_with_indices_mul_relu_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3891200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 19456)
    x1 = (xindex // 64) % 304
    x0 = xindex % 64
    x4 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 400, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tl.full([1], 608, tl.int64)
    tmp9 = tmp6 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = tmp5 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-38976) + x0 + (128*x1) + (77824*x2)), tmp11, other=0.0)
    tmp13 = tl.where(tmp11, tmp12, float("-inf"))
    tmp14 = 2*x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp8
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-38912) + x0 + (128*x1) + (77824*x2)), tmp18, other=0.0)
    tmp20 = tl.where(tmp18, tmp19, float("-inf"))
    tmp21 = triton_helpers.maximum(tmp20, tmp13)
    tmp22 = 1 + (2*x1)
    tmp23 = tmp22 >= tmp1
    tmp24 = tmp22 < tmp8
    tmp25 = tmp23 & tmp24
    tmp26 = tmp5 & tmp25
    tmp27 = tl.load(in_ptr0 + ((-38848) + x0 + (128*x1) + (77824*x2)), tmp26, other=0.0)
    tmp28 = tl.where(tmp26, tmp27, float("-inf"))
    tmp29 = triton_helpers.maximum(tmp28, tmp21)
    tmp30 = 2*x2
    tmp31 = tmp30 >= tmp1
    tmp32 = tmp30 < tmp3
    tmp33 = tmp31 & tmp32
    tmp34 = tmp33 & tmp10
    tmp35 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (77824*x2)), tmp34, other=0.0)
    tmp36 = tl.where(tmp34, tmp35, float("-inf"))
    tmp37 = triton_helpers.maximum(tmp36, tmp29)
    tmp38 = tmp33 & tmp17
    tmp39 = tl.load(in_ptr0 + (x0 + (128*x1) + (77824*x2)), tmp38, other=0.0)
    tmp40 = tl.where(tmp38, tmp39, float("-inf"))
    tmp41 = triton_helpers.maximum(tmp40, tmp37)
    tmp42 = tmp33 & tmp25
    tmp43 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (77824*x2)), tmp42, other=0.0)
    tmp44 = tl.where(tmp42, tmp43, float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp41)
    tmp46 = 1 + (2*x2)
    tmp47 = tmp46 >= tmp1
    tmp48 = tmp46 < tmp3
    tmp49 = tmp47 & tmp48
    tmp50 = tmp49 & tmp10
    tmp51 = tl.load(in_ptr0 + (38848 + x0 + (128*x1) + (77824*x2)), tmp50, other=0.0)
    tmp52 = tl.where(tmp50, tmp51, float("-inf"))
    tmp53 = triton_helpers.maximum(tmp52, tmp45)
    tmp54 = tmp49 & tmp17
    tmp55 = tl.load(in_ptr0 + (38912 + x0 + (128*x1) + (77824*x2)), tmp54, other=0.0)
    tmp56 = tl.where(tmp54, tmp55, float("-inf"))
    tmp57 = triton_helpers.maximum(tmp56, tmp53)
    tmp58 = tmp49 & tmp25
    tmp59 = tl.load(in_ptr0 + (38976 + x0 + (128*x1) + (77824*x2)), tmp58, other=0.0)
    tmp60 = tl.where(tmp58, tmp59, float("-inf"))
    tmp61 = triton_helpers.maximum(tmp60, tmp57)
    tl.store(out_ptr0 + (x4), tmp61, None)
''')
