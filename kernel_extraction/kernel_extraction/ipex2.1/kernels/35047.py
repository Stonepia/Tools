

# Original file: ./pyhpc_isoneutral_mixing___60.0/pyhpc_isoneutral_mixing___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ex/cexksy36e2zivmhdvga7oxsogau4f3zc5ye2thssdsogmrz3uthc.py
# Source Nodes: [iadd_9], Original ATen: [aten.slice_scatter]
# iadd_9 => slice_scatter_144
triton_poi_fused_slice_scatter_11 = async_compile.triton('triton_poi_fused_slice_scatter_11', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*fp64', 2: '*fp64', 3: '*fp64', 4: '*fp64', 5: '*fp64', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]})
@triton.jit
def triton_poi_fused_slice_scatter_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x3 = (xindex // 26)
    x2 = (xindex // 5200)
    x1 = (xindex // 26) % 200
    x4 = xindex % 5200
    x5 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 25, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (25*x3)), tmp2 & xmask, other=0.0)
    tmp4 = tl.where(tmp2, tmp3, 0.0)
    tmp5 = 2 + x2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 202, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = 2 + x1
    tmp12 = tmp11 >= tmp6
    tmp13 = tmp11 < tmp8
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp10
    tmp16 = tmp2 & tmp15
    tmp17 = tl.load(in_ptr1 + (1 + x2), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp18 = tl.load(in_ptr2 + (10660 + x4 + (5304*x2)), tmp16 & xmask, other=0.0)
    tmp19 = tmp17 * tmp18
    tmp20 = tl.load(in_ptr3 + (x0 + (25*x3)), tmp16 & xmask, other=0.0)
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
    tmp32 = tmp20 * tmp20
    tmp33 = tmp31 * tmp32
    tmp34 = tl.load(in_ptr4 + (10660 + x4 + (5304*x2)), tmp16 & xmask, other=0.0)
    tmp35 = tmp33 * tmp34
    tmp36 = tl.full([1], 0.0, tl.float64)
    tmp37 = tmp36 + tmp35
    tmp38 = tl.where(tmp16, tmp37, 0.0)
    tmp39 = tl.where(tmp2, tmp38, tmp36)
    tmp40 = tl.where(tmp15, tmp39, 0.0)
    tmp41 = tl.where(tmp14, tmp40, tmp36)
    tmp42 = tl.where(tmp10, tmp41, 0.0)
    tmp43 = tl.where(tmp10, tmp42, tmp36)
    tmp44 = tl.where(tmp2, tmp4, tmp43)
    tl.store(out_ptr0 + (x5), tmp44, xmask)
''')
