

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/m3/cm34g2r3vdbpjl5kmv3pnhoopcyf2kzfnnoi3r7lqtwtsdchk57w.py
# Source Nodes: [sub_26], Original ATen: [aten.sub]
# sub_26 => sub_26
triton_poi_fused_sub_62 = async_compile.triton('triton_poi_fused_sub_62', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[2097152], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_62', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_poi_fused_sub_62(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1076712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 5304)
    x4 = xindex
    x1 = (xindex // 26) % 204
    x3 = xindex % 5304
    tmp23 = tl.load(in_ptr2 + (15912 + (3*x4)), xmask).to(tl.float32)
    tmp40 = tl.load(in_ptr2 + (3*x4), xmask).to(tl.float32)
    tmp0 = 1 + x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 202, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + ((-15912) + (3*x4)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp7 = tl.where(tmp5, tmp6, 0.0)
    tmp8 = x1
    tmp9 = tmp8 >= tmp1
    tmp10 = tmp8 < tmp3
    tmp11 = tmp9 & tmp10
    tmp12 = tmp11 & tmp5
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tl.full([1], 1, tl.int32)
    tmp15 = tmp13 == tmp14
    tmp16 = tl.load(in_ptr1 + ((-5252) + x3 + (5200*x2)), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (15912 + (3*x4)), tmp12 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.where(tmp15, tmp16, tmp17)
    tmp19 = tl.where(tmp12, tmp18, 0.0)
    tmp20 = tl.load(in_ptr2 + (15912 + (3*x4)), tmp5 & xmask, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp11, tmp19, tmp20)
    tmp22 = tl.where(tmp5, tmp21, 0.0)
    tmp24 = tl.where(tmp5, tmp22, tmp23)
    tmp25 = tl.where(tmp5, tmp7, tmp24)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tl.load(in_ptr0 + ((-31824) + (3*x4)), tmp29 & xmask, other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp29, tmp30, 0.0)
    tmp32 = tmp11 & tmp29
    tmp33 = tl.load(in_ptr1 + ((-10452) + x3 + (5200*x2)), tmp32 & xmask, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr2 + (3*x4), tmp32 & xmask, other=0.0).to(tl.float32)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp32, tmp35, 0.0)
    tmp37 = tl.load(in_ptr2 + (3*x4), tmp29 & xmask, other=0.0).to(tl.float32)
    tmp38 = tl.where(tmp11, tmp36, tmp37)
    tmp39 = tl.where(tmp29, tmp38, 0.0)
    tmp41 = tl.where(tmp29, tmp39, tmp40)
    tmp42 = tl.where(tmp29, tmp31, tmp41)
    tmp43 = tmp25 - tmp42
    tl.store(out_ptr0 + (x4), tmp43, xmask)
''')
