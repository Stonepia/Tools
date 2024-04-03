

# Original file: ./resmlp_12_224___60.0/resmlp_12_224___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/float32/7i/c7iei35z7y23z3gig5oef2a6cz72ufno4h6syedkal7qyeauekxy.py
# Source Nodes: [add_22, add_23, addcmul_24, getattr_l__mod___blocks___11___mlp_channels_drop2, mean, mul_22, mul_23], Original ATen: [aten.add, aten.addcmul, aten.clone, aten.mean, aten.mul]
# add_22 => add_67
# add_23 => add_71
# addcmul_24 => add_72, mul_109
# getattr_l__mod___blocks___11___mlp_channels_drop2 => clone_35
# mean => mean
# mul_22 => mul_101
# mul_23 => mul_107
triton_red_fused_add_addcmul_clone_mean_mul_7 = async_compile.triton('triton_red_fused_add_addcmul_clone_mean_mul_7', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[65536, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_addcmul_clone_mean_mul_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_addcmul_clone_mean_mul_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 196
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    x1 = (xindex // 384)
    tmp3 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    x3 = xindex
    tmp7 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp2 = tl.load(in_ptr2 + (x0 + (384*r2) + (75264*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr4 + (r2 + (196*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.load(in_ptr6 + (x0 + (384*r2) + (75264*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 * tmp4
        tmp6 = tmp2 + tmp5
        tmp9 = tmp7 * tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tmp1 * tmp10
        tmp12 = tmp0 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp16 = 196.0
    tmp17 = tmp14 / tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')
