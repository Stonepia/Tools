

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/j7/cj7uizcvu2sw4rnrwex6qpqcsa7kabrimayoc6brplleinoak47s.py
# Source Nodes: [and__1, ge_1, setitem_82, setitem_83, setitem_84, where_2], Original ATen: [aten.bitwise_and, aten.copy, aten.ge, aten.select_scatter, aten.where]
# and__1 => bitwise_and_1
# ge_1 => ge_1
# setitem_82 => copy_82, select_scatter_128
# setitem_83 => copy_83, select_scatter_129
# setitem_84 => copy_84
# where_2 => where_2
triton_poi_fused_bitwise_and_copy_ge_select_scatter_where_59 = async_compile.triton('triton_poi_fused_bitwise_and_copy_ge_select_scatter_where_59', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bitwise_and_copy_ge_select_scatter_where_59', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_bitwise_and_copy_ge_select_scatter_where_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 26) % 200
    x2 = (xindex // 5200)
    x0 = xindex % 26
    x3 = (xindex // 26)
    x4 = xindex
    x5 = xindex % 5200
    tmp0 = tl.load(in_ptr0 + (410 + x1 + (204*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x3), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x4), xmask).to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (31981 + (3*x5) + (15912*x2)), xmask).to(tl.float32)
    tmp1 = tl.full([1], 1, tl.int64)
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 >= tmp3
    tmp5 = x0
    tmp6 = tmp5 >= tmp2
    tmp7 = tmp4 & tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = tmp5 == tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp5 == tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tl.where(tmp13, tmp15, tmp16)
    tmp18 = tl.where(tmp9, tmp11, tmp17)
    tmp20 = tl.where(tmp7, tmp18, tmp19)
    tl.store(in_out_ptr0 + (x4), tmp20, xmask)
''')
