

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/nf/cnfc5po7hy6ksz3dcq342jggu4hnmdkpexiyqg3z5yidleoesfli.py
# Source Nodes: [sub_49], Original ATen: [aten.sub]
# sub_49 => sub_49
triton_poi_fused_sub_84 = async_compile.triton('triton_poi_fused_sub_84', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_84', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused_sub_84(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 204)
    x0 = xindex % 204
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp9 & tmp5
    tmp11 = tl.full([1], 25, tl.int64)
    tmp12 = tmp11 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + ((-10025) + (25*x0) + (5000*x1)), tmp13 & xmask, other=0.0)
    tmp15 = tl.where(tmp13, tmp14, 0.0)
    tmp16 = 0.0
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp10, tmp17, 0.0)
    tmp19 = tl.where(tmp9, tmp18, tmp16)
    tmp20 = tl.where(tmp5, tmp19, 0.0)
    tmp21 = tl.where(tmp5, tmp20, tmp16)
    tmp22 = tl.full([1], 24, tl.int64)
    tmp23 = tmp22 < tmp11
    tmp24 = tmp23 & tmp10
    tmp25 = tl.load(in_ptr0 + ((-10026) + (25*x0) + (5000*x1)), tmp24 & xmask, other=0.0)
    tmp26 = tl.where(tmp24, tmp25, 0.0)
    tmp27 = tl.where(tmp23, tmp26, tmp16)
    tmp28 = tl.where(tmp10, tmp27, 0.0)
    tmp29 = tl.where(tmp9, tmp28, tmp16)
    tmp30 = tl.where(tmp5, tmp29, 0.0)
    tmp31 = tl.where(tmp5, tmp30, tmp16)
    tmp32 = tmp21 - tmp31
    tl.store(out_ptr0 + (x2), tmp32, xmask)
''')
