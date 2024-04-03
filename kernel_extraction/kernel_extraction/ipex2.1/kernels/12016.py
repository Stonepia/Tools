

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/u4/cu4q3ny6vlyengqb4fn4xyltgufglh2rsy7xcbanypbtuqcslu56.py
# Source Nodes: [maximum_1, setitem_86, tensor], Original ATen: [aten.copy, aten.lift_fresh, aten.maximum, aten.select_scatter]
# maximum_1 => maximum_1
# setitem_86 => copy_86, select_scatter_131
# tensor => full_default
triton_poi_fused_copy_lift_fresh_maximum_select_scatter_60 = async_compile.triton('triton_poi_fused_copy_lift_fresh_maximum_select_scatter_60', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_lift_fresh_maximum_select_scatter_60', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_lift_fresh_maximum_select_scatter_60(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 120000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x2 = (xindex // 600)
    x1 = (xindex // 3) % 200
    x4 = (xindex // 3)
    x5 = xindex
    tmp22 = tl.load(in_ptr1 + (32056 + (78*x1) + (15912*x2)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr1 + (32055 + x0 + (78*x1) + (15912*x2)), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = 2 + x2
    tmp4 = tl.full([1], 2, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 202, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = tmp5 & tmp7
    tmp9 = 2 + x1
    tmp10 = tmp9 >= tmp4
    tmp11 = tmp9 < tmp6
    tmp12 = tmp10 & tmp11
    tmp13 = tmp12 & tmp8
    tmp14 = tmp1 == tmp1
    tmp15 = tl.load(in_ptr0 + (25 + (26*x4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr1 + (32056 + (78*x1) + (15912*x2)), tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp13, tmp17, 0.0)
    tmp19 = tl.load(in_ptr1 + (32056 + (78*x1) + (15912*x2)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tl.where(tmp12, tmp18, tmp19)
    tmp21 = tl.where(tmp8, tmp20, 0.0)
    tmp23 = tl.where(tmp8, tmp21, tmp22)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = 0.0
    tmp26 = triton_helpers.maximum(tmp25, tmp24)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.load(in_ptr1 + (32055 + x0 + (78*x1) + (15912*x2)), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.where(tmp2, tmp15, tmp28)
    tmp30 = tl.where(tmp13, tmp29, 0.0)
    tmp31 = tl.load(in_ptr1 + (32055 + x0 + (78*x1) + (15912*x2)), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp32 = tl.where(tmp12, tmp30, tmp31)
    tmp33 = tl.where(tmp8, tmp32, 0.0)
    tmp35 = tl.where(tmp8, tmp33, tmp34)
    tmp36 = tl.where(tmp2, tmp27, tmp35)
    tl.store(out_ptr0 + (x5), tmp36, xmask)
''')
