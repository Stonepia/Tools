

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/f2/cf2b6azlikbai7byqas4wxwuitunn47sldc7kr66nwtm63gu62xy.py
# Source Nodes: [setitem_8], Original ATen: [aten.slice_scatter]
# setitem_8 => slice_scatter_29, slice_scatter_30, slice_scatter_31
triton_poi_fused_slice_scatter_15 = async_compile.triton('triton_poi_fused_slice_scatter_15', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_15', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1066104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 204
    x2 = (xindex // 5304)
    x3 = xindex % 5304
    x0 = xindex % 26
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-52) + x3 + (5200*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = 1 + x2
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 >= tmp9
    tmp11 = tmp8 < tmp3
    tmp12 = tmp10 & tmp11
    tmp13 = tmp5 & tmp12
    tmp14 = x0
    tmp15 = tmp14 >= tmp9
    tmp16 = tmp15 & tmp13
    tmp17 = tl.load(in_ptr1 + ((-1) + x0), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr2 + (5304 + x5), tmp16 & xmask, other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr3 + (5304 + x5), tmp16 & xmask, other=0.0)
    tmp21 = tl.load(in_ptr4 + ((-51) + x0 + (25*x1) + (5000*x2)), tmp16 & xmask, other=0.0)
    tmp22 = tl.abs(tmp21)
    tmp23 = -tmp22
    tmp24 = 0.001
    tmp25 = tmp23 + tmp24
    tmp26 = tmp25 / tmp24
    tmp27 = libdevice.tanh(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = 0.5
    tmp31 = tmp29 * tmp30
    tmp32 = tmp20 * tmp31
    tmp33 = 50.0
    tmp34 = triton_helpers.maximum(tmp33, tmp32)
    tmp35 = tmp19 * tmp34
    tmp36 = 0.0
    tmp37 = tmp36 + tmp35
    tmp38 = tl.where(tmp16, tmp37, 0.0)
    tmp39 = tl.where(tmp15, tmp38, tmp36)
    tmp40 = tl.where(tmp13, tmp39, 0.0)
    tmp41 = tl.where(tmp5, tmp40, tmp36)
    tmp42 = tl.where(tmp12, tmp41, 0.0)
    tmp43 = tl.where(tmp12, tmp42, tmp36)
    tmp44 = tl.where(tmp5, tmp7, tmp43)
    tl.store(out_ptr0 + (x5), tmp44, xmask)
''')
