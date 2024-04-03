

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/xf/cxfrvlxwhckr3mt4zkpqzvo5a47bflbw2etxnhl3mncwxlgqa24q.py
# Source Nodes: [l1_loss_fn, l1_loss_fn_1, l1_loss_fn_2], Original ATen: [aten.abs, aten.mean, aten.sub]
# l1_loss_fn => abs_1, mean, sub_109
# l1_loss_fn_1 => abs_2, mean_2, sub_111
# l1_loss_fn_2 => abs_3, mean_3, sub_112
triton_red_fused_abs_mean_sub_49 = async_compile.triton('triton_red_fused_abs_mean_sub_49', '''
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
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_abs_mean_sub_49', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_abs_mean_sub_49(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 273
    rnumel = 8170
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = r1 + (8170*x0)
        tmp1 = tl.full([1, 1], 2230272, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((r1 + (8170*x0)) % 2230272), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + ((r1 + (8170*x0)) % 2230272), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tmp3 - tmp4
        tmp6 = tl.abs(tmp5)
        tmp7 = tl.where(tmp2, tmp6, 0)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask & xmask, tmp10, _tmp9)
        tmp11 = tl.load(in_ptr2 + ((2478080*(((r1 + (8170*x0)) // 371712) % 6)) + ((r1 + (8170*x0)) % 371712)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tmp11 - tmp4
        tmp13 = tl.abs(tmp12)
        tmp14 = tl.where(tmp2, tmp13, 0)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr3 + ((2478080*(((r1 + (8170*x0)) // 371712) % 6)) + ((r1 + (8170*x0)) % 371712)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tmp18 - tmp4
        tmp20 = tl.abs(tmp19)
        tmp21 = tl.where(tmp2, tmp20, 0)
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp9, xmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp16, xmask)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp23, xmask)
''')
