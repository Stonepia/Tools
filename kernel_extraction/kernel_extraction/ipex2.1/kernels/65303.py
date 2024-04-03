

# Original file: ./opacus_cifar10___60.0/opacus_cifar10___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/q2/cq2kxvotuml3mzuleykcvf6rnt3gvkpmnzd5ix3eofjeuggcme2d.py
# Source Nodes: [l__self___maxpool], Original ATen: [aten.max_pool2d_with_indices]
# l__self___maxpool => getitem
triton_poi_fused_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_0', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 8) % 8
    x0 = xindex % 8
    x3 = (xindex // 8)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + (2*x0) + (32*x3)), tmp10, other=0.0).to(tl.float32)
    tmp12 = triton_helpers.maximum(0, tmp11)
    tmp13 = tl.where(tmp10, tmp12, float("-inf"))
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-16) + (2*x0) + (32*x3)), tmp18, other=0.0).to(tl.float32)
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.where(tmp18, tmp20, float("-inf"))
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-15) + (2*x0) + (32*x3)), tmp27, other=0.0).to(tl.float32)
    tmp29 = triton_helpers.maximum(0, tmp28)
    tmp30 = tl.where(tmp27, tmp29, float("-inf"))
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (32*x3)), tmp36, other=0.0).to(tl.float32)
    tmp38 = triton_helpers.maximum(0, tmp37)
    tmp39 = tl.where(tmp36, tmp38, float("-inf"))
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (32*x3)), tmp41, other=0.0).to(tl.float32)
    tmp43 = triton_helpers.maximum(0, tmp42)
    tmp44 = tl.where(tmp41, tmp43, float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x3)), tmp46, other=0.0).to(tl.float32)
    tmp48 = triton_helpers.maximum(0, tmp47)
    tmp49 = tl.where(tmp46, tmp48, float("-inf"))
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (15 + (2*x0) + (32*x3)), tmp55, other=0.0).to(tl.float32)
    tmp57 = triton_helpers.maximum(0, tmp56)
    tmp58 = tl.where(tmp55, tmp57, float("-inf"))
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x3)), tmp60, other=0.0).to(tl.float32)
    tmp62 = triton_helpers.maximum(0, tmp61)
    tmp63 = tl.where(tmp60, tmp62, float("-inf"))
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x3)), tmp65, other=0.0).to(tl.float32)
    tmp67 = triton_helpers.maximum(0, tmp66)
    tmp68 = tl.where(tmp65, tmp67, float("-inf"))
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (x4), tmp69, None)
''')
