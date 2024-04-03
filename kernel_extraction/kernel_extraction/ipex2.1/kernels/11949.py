

# Original file: ./pyhpc_turbulent_kinetic_energy___60.0/pyhpc_turbulent_kinetic_energy___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/kv/ckv7h5iwjawa5ob4ogo3tggsttrrqcacgbqlgm3hs5io3psynd7f.py
# Source Nodes: [iadd_53, neg_56, sub_48, truediv_76], Original ATen: [aten.add, aten.div, aten.neg, aten.sub]
# iadd_53 => add_69
# neg_56 => neg_56
# sub_48 => sub_48
# truediv_76 => div_73
triton_poi_fused_add_div_neg_sub_77 = async_compile.triton('triton_poi_fused_add_div_neg_sub_77', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_neg_sub_77', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_div_neg_sub_77(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998784
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x3 = (xindex // 24)
    x2 = (xindex // 4896)
    x1 = (xindex // 24) % 204
    x4 = xindex
    tmp3 = tl.load(in_ptr0 + (3*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (3*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (3 + (3*x0) + (78*x3)), xmask).to(tl.float32)
    tmp37 = tl.load(in_ptr5 + (1 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = 1 + x0
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp5 = x2
    tmp6 = tl.full([1], 2, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 202, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp7 & tmp9
    tmp11 = x1
    tmp12 = tmp11 >= tmp6
    tmp13 = tmp11 < tmp8
    tmp14 = tmp12 & tmp13
    tmp15 = tmp14 & tmp10
    tmp16 = tmp1 == tmp1
    tmp17 = tl.load(in_ptr2 + ((-10451) + x0 + (26*x1) + (5200*x2)), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (3 + (3*x0) + (78*x3)), tmp15 & xmask, other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp16, tmp17, tmp18)
    tmp20 = tl.where(tmp15, tmp19, 0.0)
    tmp21 = tl.load(in_ptr3 + (3 + (3*x0) + (78*x3)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp22 = tl.where(tmp14, tmp20, tmp21)
    tmp23 = tl.where(tmp10, tmp22, 0.0)
    tmp25 = tl.where(tmp10, tmp23, tmp24)
    tmp26 = tl.where(tmp2, tmp4, tmp25)
    tmp27 = tl.where(tmp2, tmp3, tmp26)
    tmp28 = tl.load(in_ptr4 + ((-10607) + x0 + (26*x3)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp29 = tl.where(tmp10, tmp28, 0.0)
    tmp30 = 0.0
    tmp31 = tl.where(tmp10, tmp29, tmp30)
    tmp32 = tl.load(in_ptr4 + ((-10608) + x0 + (26*x3)), tmp10 & xmask, other=0.0).to(tl.float32)
    tmp33 = tl.where(tmp10, tmp32, 0.0)
    tmp34 = tl.where(tmp10, tmp33, tmp30)
    tmp35 = tmp31 - tmp34
    tmp36 = -tmp35
    tmp38 = tmp36 / tmp37
    tmp39 = tmp27 + tmp38
    tl.store(out_ptr0 + (x4), tmp39, xmask)
''')
