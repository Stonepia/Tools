

# Original file: ./timm_regnet___60.0/timm_regnet___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/yg/cygfsucgnnlpeaii7ivzqvqirtawc7hjtykmnbjuojzonks25fiv.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[8], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'configs': [AttrsDescriptor(divisible_by_16=(0,), equal_to_1=())]})
@triton.jit
def triton_poi_fused_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = 0.0
    tmp8 = tl.where(tmp6, tmp7, tmp7)
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.where(tmp10, tmp7, tmp7)
    tmp12 = tl.where(tmp4, tmp8, tmp11)
    tmp13 = tl.full([1], 6, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.full([1], 5, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.where(tmp16, tmp7, tmp7)
    tmp18 = tl.full([1], 7, tl.int64)
    tmp19 = tmp0 < tmp18
    tmp20 = tl.where(tmp19, tmp7, tmp7)
    tmp21 = tl.where(tmp14, tmp17, tmp20)
    tmp22 = tl.where(tmp2, tmp12, tmp21)
    tl.store(out_ptr0 + (x0), tmp22, xmask)
''')
