

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/mr/cmr6hijopzkdi365srge7nvwzij36kws6kmnp3rtwcnmnvysodmr.py
# Source Nodes: [and__1, ge_1, iadd_2, mul_14, setitem_7, setitem_9], Original ATen: [aten.bitwise_and, aten.copy, aten.ge, aten.mul, aten.select_scatter, aten.slice_scatter]
# and__1 => bitwise_and_1
# ge_1 => ge_1
# iadd_2 => select_scatter_6
# mul_14 => mul_17
# setitem_7 => copy_7, select_scatter_3, slice_scatter_18, slice_scatter_19
# setitem_9 => copy_9, select_scatter_7
triton_poi_fused_bitwise_and_copy_ge_mul_select_scatter_slice_scatter_4 = async_compile.triton('triton_poi_fused_bitwise_and_copy_ge_mul_select_scatter_slice_scatter_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp64', 1: '*i64', 2: '*fp64', 3: '*fp64', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_copy_ge_mul_select_scatter_slice_scatter_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_and_copy_ge_mul_select_scatter_slice_scatter_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x3 = (xindex // 26)
    x1 = (xindex // 26) % 200
    x2 = (xindex // 5200)
    x4 = xindex
    tmp4 = tl.load(in_ptr0 + (x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (410 + x1 + (204*x2)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (25 + (26*x3)), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr2 + (1 + (26*x3)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (x4), xmask)
    tmp0 = x0
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp1 == tmp1
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = tmp6 >= tmp7
    tmp11 = tmp9 & tmp10
    tmp12 = tmp11.to(tl.float64)
    tmp13 = tl.full([1], 25, tl.int32)
    tmp14 = tmp1 == tmp13
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tmp12 * tmp17
    tmp19 = tl.where(tmp3, tmp4, tmp18)
    tmp20 = tmp0 >= tmp7
    tmp21 = tmp9 & tmp20
    tmp22 = tmp21.to(tl.float64)
    tmp23 = tmp0 == tmp13
    tmp25 = tl.where(tmp23, tmp15, tmp24)
    tmp26 = tmp22 * tmp25
    tmp27 = tl.where(tmp2, tmp4, tmp26)
    tmp28 = tl.where(tmp2, tmp19, tmp27)
    tl.store(out_ptr0 + (x4), tmp28, xmask)
''')
