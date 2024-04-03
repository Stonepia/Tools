

# Original file: ./inception_v3___60.0/inception_v3___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_bf16/6x/c6xwy4kqticods5hs3uztxeoluonxxsp46r4lbgjpeydsvlqhtva.py
# Source Nodes: [l__self___pool2], Original ATen: [aten.max_pool2d_with_indices]
# l__self___pool2 => max_pool2d_with_indices_1
triton_poi_fused_max_pool2d_with_indices_2 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30105600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192) % 35
    x2 = (xindex // 6720) % 35
    x3 = (xindex // 235200)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (13632 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (13824 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (14016 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp11 = tl.load(in_ptr0 + (27264 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (27456 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (27648 + x0 + (384*x1) + (27264*x2) + (967872*x3)), None).to(tl.float32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')
