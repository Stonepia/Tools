

# Original file: ./functorch_dp_cifar10___60.0/functorch_dp_cifar10___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/lu/cludlrohghx5no2bmuz3ci3chxqgfsepfztf4ffjr2g55erl3lbw.py
# Source Nodes: [l__self___bn1, l__self___maxpool, l__self___relu], Original ATen: [aten.max_pool2d_with_indices, aten.native_group_norm, aten.relu]
# l__self___bn1 => add_1, convert_element_type_3, mul_1
# l__self___maxpool => max_pool2d_with_indices
# l__self___relu => relu
triton_poi_fused_max_pool2d_with_indices_native_group_norm_relu_3 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_native_group_norm_relu_3', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_native_group_norm_relu_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_native_group_norm_relu_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 512) % 8
    x1 = (xindex // 64) % 8
    x0 = xindex % 64
    x5 = (xindex // 512)
    x6 = xindex
    tmp0 = (-1) + (2*x2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x1)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-1088) + x0 + (128*x1) + (2048*x5)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp10, tmp11, float("-inf"))
    tmp13 = 2*x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tmp13 < tmp3
    tmp16 = tmp14 & tmp15
    tmp17 = tmp5 & tmp16
    tmp18 = tl.load(in_ptr0 + ((-1024) + x0 + (128*x1) + (2048*x5)), tmp17, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp17, tmp18, float("-inf"))
    tmp20 = triton_helpers.maximum(tmp19, tmp12)
    tmp21 = 1 + (2*x1)
    tmp22 = tmp21 >= tmp1
    tmp23 = tmp21 < tmp3
    tmp24 = tmp22 & tmp23
    tmp25 = tmp5 & tmp24
    tmp26 = tl.load(in_ptr0 + ((-960) + x0 + (128*x1) + (2048*x5)), tmp25, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp25, tmp26, float("-inf"))
    tmp28 = triton_helpers.maximum(tmp27, tmp20)
    tmp29 = 2*x2
    tmp30 = tmp29 >= tmp1
    tmp31 = tmp29 < tmp3
    tmp32 = tmp30 & tmp31
    tmp33 = tmp32 & tmp9
    tmp34 = tl.load(in_ptr0 + ((-64) + x0 + (128*x1) + (2048*x5)), tmp33, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp33, tmp34, float("-inf"))
    tmp36 = triton_helpers.maximum(tmp35, tmp28)
    tmp37 = tmp32 & tmp16
    tmp38 = tl.load(in_ptr0 + (x0 + (128*x1) + (2048*x5)), tmp37, other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp37, tmp38, float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp36)
    tmp41 = tmp32 & tmp24
    tmp42 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (2048*x5)), tmp41, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, float("-inf"))
    tmp44 = triton_helpers.maximum(tmp43, tmp40)
    tmp45 = 1 + (2*x2)
    tmp46 = tmp45 >= tmp1
    tmp47 = tmp45 < tmp3
    tmp48 = tmp46 & tmp47
    tmp49 = tmp48 & tmp9
    tmp50 = tl.load(in_ptr0 + (960 + x0 + (128*x1) + (2048*x5)), tmp49, other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp49, tmp50, float("-inf"))
    tmp52 = triton_helpers.maximum(tmp51, tmp44)
    tmp53 = tmp48 & tmp16
    tmp54 = tl.load(in_ptr0 + (1024 + x0 + (128*x1) + (2048*x5)), tmp53, other=0.0).to(tl.float32)
    tmp55 = tl.where(tmp53, tmp54, float("-inf"))
    tmp56 = triton_helpers.maximum(tmp55, tmp52)
    tmp57 = tmp48 & tmp24
    tmp58 = tl.load(in_ptr0 + (1088 + x0 + (128*x1) + (2048*x5)), tmp57, other=0.0).to(tl.float32)
    tmp59 = tl.where(tmp57, tmp58, float("-inf"))
    tmp60 = triton_helpers.maximum(tmp59, tmp56)
    tl.store(out_ptr0 + (x6), tmp60, None)
''')
