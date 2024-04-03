

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/bs/cbs7iih763jdbnrrlvxikt4wnza2pc3or3ka2uicvff6uxdbrpn7.py
# Source Nodes: [sub_48], Original ATen: [aten.sub]
# sub_48 => sub_48
triton_poi_fused_sub_82 = async_compile.triton('triton_poi_fused_sub_82', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_82', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_sub_82(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 4896)
    x1 = (xindex // 24) % 204
    x0 = xindex % 24
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = 1 + x0
    tmp12 = tl.full([1], 25, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tmp13 & tmp10
    tmp15 = tl.load(in_ptr0 + ((-10049) + x0 + (25*x1) + (5000*x2)), tmp14 & xmask, other=0.0)
    tmp16 = tl.where(tmp14, tmp15, 0.0)
    tmp17 = 0.0
    tmp18 = tl.where(tmp13, tmp16, tmp17)
    tmp19 = tl.where(tmp10, tmp18, 0.0)
    tmp20 = tl.where(tmp9, tmp19, tmp17)
    tmp21 = tl.where(tmp5, tmp20, 0.0)
    tmp22 = tl.where(tmp5, tmp21, tmp17)
    tmp23 = x0
    tmp24 = tmp23 < tmp12
    tmp25 = tmp24 & tmp10
    tmp26 = tl.load(in_ptr0 + ((-10050) + x0 + (25*x1) + (5000*x2)), tmp25 & xmask, other=0.0)
    tmp27 = tl.where(tmp25, tmp26, 0.0)
    tmp28 = tl.where(tmp24, tmp27, tmp17)
    tmp29 = tl.where(tmp10, tmp28, 0.0)
    tmp30 = tl.where(tmp9, tmp29, tmp17)
    tmp31 = tl.where(tmp5, tmp30, 0.0)
    tmp32 = tl.where(tmp5, tmp31, tmp17)
    tmp33 = tmp22 - tmp32
    tl.store(out_ptr0 + (x4), tmp33, xmask)
''')
