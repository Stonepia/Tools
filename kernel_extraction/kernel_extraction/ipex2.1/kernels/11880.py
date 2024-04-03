

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/sc/cscowciquxoiagevoioi6ljngycv6wrmd2d6f375zjjdpxzh3gp4.py
# Source Nodes: [iadd_4, iadd_6, setitem_11, setitem_9], Original ATen: [aten._to_copy, aten.add, aten.copy, aten.select_scatter]
# iadd_4 => add_13, convert_element_type_1, select_scatter_10
# iadd_6 => select_scatter_14
# setitem_11 => copy_11, select_scatter_11
# setitem_9 => copy_9, select_scatter_7
triton_poi_fused__to_copy_add_copy_select_scatter_8 = async_compile.triton('triton_poi_fused__to_copy_add_copy_select_scatter_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_copy_select_scatter_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy_add_copy_select_scatter_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1040000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 26
    x1 = (xindex // 26)
    x2 = xindex
    tmp3 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (1 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr1 + (2 + (26*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp0 = x0
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 2, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp6 = tmp4 == tmp4
    tmp7 = tl.full([1], 1, tl.int32)
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
