

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/3c/c3cuc4auloncggloakcsca6uevmeosvi55rz7qf2xh4xakst7pgf.py
# Source Nodes: [setitem_15], Original ATen: [aten.select_scatter]
# setitem_15 => select_scatter_8
triton_poi_fused_select_scatter_6 = async_compile.triton('triton_poi_fused_select_scatter_6', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_scatter_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_select_scatter_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4180800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 2) % 2
    x0 = xindex % 2
    x5 = (xindex // 4)
    x4 = (xindex // 20800)
    x3 = (xindex // 104) % 200
    x9 = (xindex // 4) % 5200
    x6 = xindex % 20800
    x10 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + (2*x5)), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr3 + (21424 + x6 + (21216*x4)), xmask)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = 1 + x4
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 202, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = 2 + x3
    tmp11 = tl.full([1], 2, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = tmp10 < tmp7
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp9
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tmp0 == tmp16
    tmp18 = x0
    tmp19 = tmp18 == tmp1
    tmp20 = tl.load(in_ptr1 + (x5), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.abs(tmp20)
    tmp22 = -tmp21
    tmp23 = tl.full([1], 0.001, tl.float64)
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp23
    tmp26 = libdevice.tanh(tmp25)
    tmp27 = tl.full([1], 1.0, tl.float64)
    tmp28 = tmp26 + tmp27
    tmp29 = tl.full([1], 0.5, tl.float64)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30 * tmp20
    tmp32 = tl.load(in_ptr2 + (5356 + x9 + (5304*x4)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 * tmp32
    tmp34 = tl.load(in_ptr3 + (21424 + x0 + (4*x9) + (21216*x4)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.load(in_ptr3 + (21424 + x6 + (21216*x4)), tmp15 & xmask, other=0.0)
    tmp37 = tl.where(tmp17, tmp35, tmp36)
    tmp38 = tl.where(tmp15, tmp37, 0.0)
    tmp39 = tl.load(in_ptr3 + (21424 + x6 + (21216*x4)), tmp9 & xmask, other=0.0)
    tmp40 = tl.where(tmp14, tmp38, tmp39)
    tmp41 = tl.where(tmp9, tmp40, 0.0)
    tmp43 = tl.where(tmp9, tmp41, tmp42)
    tmp44 = tl.where(tmp2, tmp3, tmp43)
    tl.store(out_ptr0 + (x10), tmp44, xmask)
''')
