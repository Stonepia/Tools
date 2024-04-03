

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/vh/cvhmcujs6lrju2azlkvj4dex5dmwkal3shsvxe26hdr4nsdggmp3.py
# Source Nodes: [iadd_52, neg_55, truediv_75], Original ATen: [aten.add, aten.div, aten.neg, aten.select_scatter]
# iadd_52 => add_68, select_scatter_138
# neg_55 => neg_55
# truediv_75 => div_72
triton_poi_fused_add_div_neg_select_scatter_82 = async_compile.triton('triton_poi_fused_add_div_neg_select_scatter_82', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_neg_select_scatter_82', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_neg_select_scatter_82(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 124848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x2 = (xindex // 612)
    x1 = (xindex // 3) % 204
    x3 = (xindex // 3)
    x4 = xindex
    tmp22 = tl.load(in_ptr1 + (78*x3), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (0))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp39 = tl.load(in_ptr1 + (x0 + (78*x3)), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 202, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = x1
    tmp10 = tmp9 >= tmp4
    tmp11 = tmp9 < tmp6
    tmp12 = tmp10 & tmp11
    tmp13 = tmp12 & tmp8
    tmp14 = tmp1 == tmp1
    tmp15 = tl.load(in_ptr0 + ((-10452) + (26*x1) + (5200*x2)), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr1 + (78*x3), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp13, tmp17, 0.0)
    tmp19 = tl.load(in_ptr1 + (78*x3), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp12, tmp18, tmp19)
    tmp21 = tl.where(tmp8, tmp20, 0.0)
    tmp23 = tl.where(tmp8, tmp21, tmp22)
    tmp24 = tl.load(in_ptr2 + ((-10608) + (26*x3)), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp8, tmp24, 0.0)
    tmp26 = tl.full([1], 0.0, tl.float64)
    tmp27 = tl.where(tmp8, tmp25, tmp26)
    tmp28 = -tmp27
    tmp31 = tmp28 / tmp30
    tmp32 = tmp23 + tmp31
    tmp33 = tl.load(in_ptr1 + (x0 + (78*x3)), tmp13 & xmask, other=0.0)
    tmp34 = tl.where(tmp2, tmp15, tmp33)
    tmp35 = tl.where(tmp13, tmp34, 0.0)
    tmp36 = tl.load(in_ptr1 + (x0 + (78*x3)), tmp8 & xmask, other=0.0)
    tmp37 = tl.where(tmp12, tmp35, tmp36)
    tmp38 = tl.where(tmp8, tmp37, 0.0)
    tmp40 = tl.where(tmp8, tmp38, tmp39)
    tmp41 = tl.where(tmp2, tmp32, tmp40)
    tl.store(out_ptr0 + (x4), tmp41, xmask)
''')
