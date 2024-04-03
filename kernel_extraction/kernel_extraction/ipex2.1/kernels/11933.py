

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/g4/cg4wdd2fjvsvwq3ktcchpyw2oowco5lq7nlyoofi2dcmgar5olhf.py
# Source Nodes: [setitem_86], Original ATen: [aten.select_scatter, aten.slice_scatter]
# setitem_86 => select_scatter_132, slice_scatter_25
triton_poi_fused_select_scatter_slice_scatter_61 = async_compile.triton('triton_poi_fused_select_scatter_slice_scatter_61', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_select_scatter_slice_scatter_61', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_select_scatter_slice_scatter_61(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3182400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 78) % 204
    x1 = (xindex // 3) % 26
    x0 = xindex % 3
    x3 = (xindex // 15912)
    x5 = (xindex // 3) % 5304
    x6 = xindex
    tmp38 = tl.load(in_ptr2 + (31824 + x6), xmask).to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tl.full([1], 25, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = tl.load(in_ptr0 + ((-6) + x0 + (3*x2) + (600*x3)), tmp5 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = 2 + x3
    tmp11 = tmp10 >= tmp1
    tmp12 = tmp10 < tmp3
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13 & tmp5
    tmp15 = tmp5 & tmp14
    tmp16 = x0
    tmp17 = tl.full([1], 1, tl.int32)
    tmp18 = tmp16 == tmp17
    tmp19 = tl.load(in_ptr1 + ((-52) + x5 + (5200*x3)), tmp15 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.load(in_ptr2 + (31824 + x6), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp15, tmp21, 0.0)
    tmp23 = tl.load(in_ptr2 + (31824 + x6), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tl.where(tmp14, tmp24, 0.0)
    tmp26 = tl.load(in_ptr2 + (31824 + x6), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp13, tmp25, tmp26)
    tmp28 = tl.where(tmp8, tmp9, tmp27)
    tmp29 = tl.where(tmp5, tmp28, 0.0)
    tmp30 = tmp5 & tmp13
    tmp31 = tl.load(in_ptr1 + ((-52) + x5 + (5200*x3)), tmp30 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr2 + (31824 + x6), tmp30 & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp18, tmp31, tmp32)
    tmp34 = tl.where(tmp30, tmp33, 0.0)
    tmp35 = tl.load(in_ptr2 + (31824 + x6), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp36 = tl.where(tmp5, tmp34, tmp35)
    tmp37 = tl.where(tmp13, tmp36, 0.0)
    tmp39 = tl.where(tmp13, tmp37, tmp38)
    tmp40 = tl.where(tmp5, tmp29, tmp39)
    tl.store(out_ptr0 + (x6), tmp40, xmask)
''')
