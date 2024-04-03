

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/zz/czzglbcjkceia3glu3kahuqpsjywey5byeexrx46wtsj6vdqppon.py
# Source Nodes: [iadd_46, iadd_48, setitem_51, setitem_53], Original ATen: [aten._to_copy, aten.copy, aten.select_scatter]
# iadd_46 => convert_element_type_22, select_scatter_94
# iadd_48 => convert_element_type_23, select_scatter_98
# setitem_51 => copy_51, select_scatter_91
# setitem_53 => copy_53, select_scatter_95
triton_poi_fused__to_copy_copy_select_scatter_44 = async_compile.triton('triton_poi_fused__to_copy_copy_select_scatter_44', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_copy_select_scatter_44', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_copy_select_scatter_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (22 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (23 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 24, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.full([1], 23, tl.int32)
    tmp6 = tmp0 == tmp5
    tmp7 = tmp5 == tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.full([1], 22, tl.int32)
    tmp11 = tmp5 == tmp10
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tl.where(tmp7, tmp9, tmp14)
    tmp16 = tmp0 == tmp10
    tmp18 = tl.where(tmp16, tmp12, tmp17)
    tmp19 = tl.where(tmp6, tmp9, tmp18)
    tmp20 = tl.where(tmp6, tmp15, tmp19)
    tmp21 = tl.where(tmp2, tmp4, tmp20)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''')
