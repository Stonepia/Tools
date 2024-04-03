

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/bo/cbopb7nngzxtiyenitbqwbxyl7je3lpfunbcyeo7flmi6xagmt6x.py
# Source Nodes: [setitem_8], Original ATen: [aten.copy, aten.slice_scatter]
# setitem_8 => copy_8, slice_scatter_28
triton_poi_fused_copy_slice_scatter_14 = async_compile.triton('triton_poi_fused_copy_slice_scatter_14', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_slice_scatter_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_copy_slice_scatter_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1045200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x2 = (xindex // 5200)
    x1 = (xindex // 26) % 200
    x3 = xindex % 5200
    x4 = (xindex // 26)
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = 1 + x2
    tmp4 = tmp3 >= tmp1
    tmp5 = tl.full([1], 202, tl.int64)
    tmp6 = tmp3 < tmp5
    tmp7 = tmp4 & tmp6
    tmp8 = tmp7 & tmp2
    tmp9 = 2 + x1
    tmp10 = tl.full([1], 2, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = tmp9 < tmp5
    tmp13 = tmp11 & tmp12
    tmp14 = tmp13 & tmp8
    tmp15 = tmp2 & tmp14
    tmp16 = tl.load(in_ptr0 + ((-1) + x0), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr1 + (5356 + x3 + (5304*x2)), tmp15 & xmask, other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp15 & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + ((-1) + x0 + (25*x4)), tmp15 & xmask, other=0.0)
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
    tmp31 = tmp19 * tmp30
    tmp32 = tl.full([1], 50.0, tl.float64)
    tmp33 = triton_helpers.maximum(tmp32, tmp31)
    tmp34 = tmp18 * tmp33
    tmp35 = tl.full([1], 0.0, tl.float64)
    tmp36 = tmp35 + tmp34
    tmp37 = tl.where(tmp15, tmp36, 0.0)
    tmp38 = tl.where(tmp2, tmp37, tmp35)
    tmp39 = tl.where(tmp14, tmp38, 0.0)
    tmp40 = tl.where(tmp13, tmp39, tmp35)
    tmp41 = tl.where(tmp8, tmp40, 0.0)
    tmp42 = tl.where(tmp7, tmp41, tmp35)
    tmp43 = tl.where(tmp2, tmp42, 0.0)
    tmp44 = tmp13 & tmp7
    tmp45 = tmp2 & tmp44
    tmp46 = tl.load(in_ptr0 + ((-1) + x0), tmp45 & xmask, eviction_policy='evict_last', other=0.0)
    tmp47 = tl.load(in_ptr1 + (5356 + x3 + (5304*x2)), tmp45 & xmask, other=0.0)
    tmp48 = tmp46 * tmp47
    tmp49 = tl.load(in_ptr2 + (5356 + x3 + (5304*x2)), tmp45 & xmask, other=0.0)
    tmp50 = tl.load(in_ptr3 + ((-1) + x0 + (25*x4)), tmp45 & xmask, other=0.0)
    tmp51 = tl.abs(tmp50)
    tmp52 = -tmp51
    tmp53 = tmp52 + tmp23
    tmp54 = tmp53 / tmp23
    tmp55 = libdevice.tanh(tmp54)
    tmp56 = tmp55 + tmp27
    tmp57 = tmp56 * tmp29
    tmp58 = tmp49 * tmp57
    tmp59 = triton_helpers.maximum(tmp32, tmp58)
    tmp60 = tmp48 * tmp59
    tmp61 = tmp35 + tmp60
    tmp62 = tl.where(tmp45, tmp61, 0.0)
    tmp63 = tl.where(tmp2, tmp62, tmp35)
    tmp64 = tl.where(tmp44, tmp63, 0.0)
    tmp65 = tl.where(tmp13, tmp64, tmp35)
    tmp66 = tl.where(tmp7, tmp65, 0.0)
    tmp67 = tl.where(tmp7, tmp66, tmp35)
    tmp68 = tl.where(tmp2, tmp43, tmp67)
    tl.store(out_ptr0 + (x5), tmp68, xmask)
''')
