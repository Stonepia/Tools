

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ga/cgajsb36dzvcrlpqkjkgzeyymizlgitialjos3lq2eba4p3h3qk6.py
# Source Nodes: [setitem_91], Original ATen: [aten.copy, aten.select_scatter]
# setitem_91 => copy_91, select_scatter_136
triton_poi_fused_copy_select_scatter_66 = async_compile.triton('triton_poi_fused_copy_select_scatter_66', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_select_scatter_66', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_select_scatter_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3120000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x3 = (xindex // 15600)
    x5 = (xindex // 3) % 5200
    x2 = (xindex // 78) % 200
    x7 = (xindex // 3)
    x4 = xindex % 15600
    x8 = xindex
    tmp26 = tl.load(in_ptr3 + (31981 + (3*x5) + (15912*x3)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp40 = tl.load(in_ptr3 + (31980 + x4 + (15912*x3)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = 2 + x3
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 202, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = tl.load(in_ptr0 + (157 + (3*x5) + (15912*x3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp8, tmp9, 0.0)
    tmp11 = tl.load(in_ptr1 + (157 + (3*x5) + (15912*x3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp8, tmp11, 0.0)
    tmp13 = 2 + x2
    tmp14 = tmp13 >= tmp4
    tmp15 = tmp13 < tmp6
    tmp16 = tmp14 & tmp15
    tmp17 = tmp16 & tmp8
    tmp18 = tmp1 == tmp1
    tmp19 = tl.load(in_ptr2 + (x7), tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr3 + (31981 + (3*x5) + (15912*x3)), tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp17, tmp21, 0.0)
    tmp23 = tl.load(in_ptr3 + (31981 + (3*x5) + (15912*x3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp16, tmp22, tmp23)
    tmp25 = tl.where(tmp8, tmp24, 0.0)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = tl.where(tmp8, tmp12, tmp27)
    tmp29 = tl.where(tmp8, tmp10, tmp28)
    tmp30 = tl.load(in_ptr0 + (156 + x4 + (15912*x3)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp8, tmp30, 0.0)
    tmp32 = tl.load(in_ptr1 + (156 + x4 + (15912*x3)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp8, tmp32, 0.0)
    tmp34 = tl.load(in_ptr3 + (31980 + x4 + (15912*x3)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp2, tmp19, tmp34)
    tmp36 = tl.where(tmp17, tmp35, 0.0)
    tmp37 = tl.load(in_ptr3 + (31980 + x4 + (15912*x3)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.where(tmp16, tmp36, tmp37)
    tmp39 = tl.where(tmp8, tmp38, 0.0)
    tmp41 = tl.where(tmp8, tmp39, tmp40)
    tmp42 = tl.where(tmp8, tmp33, tmp41)
    tmp43 = tl.where(tmp8, tmp31, tmp42)
    tmp44 = tl.where(tmp2, tmp29, tmp43)
    tl.store(out_ptr0 + (x8), tmp44, xmask)
''')
