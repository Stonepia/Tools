

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/wd/cwdcpan36oppw7jj3ocfy3odkdxusqeltx3dqvz7v5irdhg5eeul.py
# Source Nodes: [iadd_16, iadd_18, setitem_21, setitem_23], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.select_scatter]
# iadd_16 => add_25, convert_element_type_7, select_scatter_34
# iadd_18 => select_scatter_38
# setitem_21 => copy_21, select_scatter_31
# setitem_23 => copy_23, select_scatter_35
triton_poi_fused__to_copy_add_copy_select_scatter_20 = async_compile.triton('triton_poi_fused__to_copy_add_copy_select_scatter_20', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_select_scatter_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_select_scatter_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (7 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (8 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 9, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 8, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = tmp4 == tmp4
    tmp7 = tl.full([1], 7, tl.int32)
    tmp8 = tmp4 == tmp7
    tmp11 = tl.where(tmp8, tmp9, tmp10)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.where(tmp6, tmp15, tmp11)
    tmp17 = tmp0 == tmp7
    tmp19 = tl.where(tmp17, tmp9, tmp18)
    tmp20 = tl.where(tmp5, tmp15, tmp19)
    tmp21 = tl.where(tmp5, tmp16, tmp20)
    tmp22 = tl.where(tmp2, tmp3, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''')
