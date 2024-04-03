

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ur/curbietxgb4xfsk7wb6avxhv2f7orowfk6ohdp4qgrylfmfvl76e.py
# Source Nodes: [setitem_38], Original ATen: [aten.copy, aten.select_scatter, aten.slice_scatter]
# setitem_38 => copy_38, select_scatter_34, slice_scatter_189
triton_poi_fused_copy_select_scatter_slice_scatter_50 = async_compile.triton('triton_poi_fused_copy_select_scatter_slice_scatter_50', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_select_scatter_slice_scatter_50', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_select_scatter_slice_scatter_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1060800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x0 = xindex % 26
    x2 = (xindex // 5304)
    x4 = xindex
    tmp52 = tl.load(in_ptr3 + (10608 + x4), xmask).to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tl.full([1], 25, tl.int32)
    tmp8 = tmp6 == tmp7
    tmp9 = 2 + x2
    tmp10 = tmp9 >= tmp1
    tmp11 = tmp9 < tmp3
    tmp12 = tmp10 & tmp11
    tmp13 = tmp12 & tmp5
    tmp14 = tmp5 & tmp13
    tmp15 = tl.full([1], 25, tl.int64)
    tmp16 = tmp6 < tmp15
    tmp17 = tmp16 & tmp14
    tmp18 = tl.load(in_ptr0 + (10608 + x4), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (2 + x2), tmp17 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = 4.0
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 / tmp21
    tmp23 = tl.load(in_ptr2 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp17 & xmask, other=0.0).to(tl.float32)
    tmp24 = tmp22 + tmp23
    tmp25 = tl.where(tmp17, tmp24, 0.0)
    tmp26 = tl.load(in_ptr3 + (10608 + x4), tmp14 & xmask, other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp16, tmp25, tmp26)
    tmp28 = tl.where(tmp14, tmp27, 0.0)
    tmp29 = tl.load(in_ptr3 + (10608 + x4), tmp13 & xmask, other=0.0).to(tl.float32)
    tmp30 = tl.where(tmp5, tmp28, tmp29)
    tmp31 = tl.where(tmp13, tmp30, 0.0)
    tmp32 = tl.load(in_ptr3 + (10608 + x4), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp12, tmp31, tmp32)
    tmp34 = 0.0
    tmp35 = tl.where(tmp8, tmp34, tmp33)
    tmp36 = tl.where(tmp5, tmp35, 0.0)
    tmp37 = tmp5 & tmp12
    tmp38 = tmp16 & tmp37
    tmp39 = tl.load(in_ptr0 + (10608 + x4), tmp38 & xmask, other=0.0).to(tl.float32)
    tmp40 = tl.load(in_ptr1 + (2 + x2), tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = tmp40 * tmp20
    tmp42 = tmp39 / tmp41
    tmp43 = tl.load(in_ptr2 + ((-50) + x0 + (25*x1) + (5000*x2)), tmp38 & xmask, other=0.0).to(tl.float32)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.where(tmp38, tmp44, 0.0)
    tmp46 = tl.load(in_ptr3 + (10608 + x4), tmp37 & xmask, other=0.0).to(tl.float32)
    tmp47 = tl.where(tmp16, tmp45, tmp46)
    tmp48 = tl.where(tmp37, tmp47, 0.0)
    tmp49 = tl.load(in_ptr3 + (10608 + x4), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp50 = tl.where(tmp5, tmp48, tmp49)
    tmp51 = tl.where(tmp12, tmp50, 0.0)
    tmp53 = tl.where(tmp12, tmp51, tmp52)
    tmp54 = tl.where(tmp5, tmp36, tmp53)
    tl.store(out_ptr0 + (x4), tmp54, xmask)
''')
