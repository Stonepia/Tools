

# Original file: ./MegatronBertForCausalLM__0_forward_277.0/MegatronBertForCausalLM__0_forward_277.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/op/cophs72d3fyuo5z5h3zzovo662w7dq3wzile7g4uouscvlvkx7gb.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => convert_element_type_466, div_48, full_default_2, full_default_3, ne, neg, sum_26, sum_27, where_1
triton_red_fused_nll_loss_backward_nll_loss_forward_20 = async_compile.triton('triton_red_fused_nll_loss_backward_nll_loss_forward_20', '''
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
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*i1', 4: '*i64', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_backward_nll_loss_forward_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_nll_loss_backward_nll_loss_forward_20(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2044
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (512*(r0 // 511)) + (r0 % 511)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tl.where(tmp2, tmp0, tmp7)
        tmp9 = tl.where(tmp8 < 0, tmp8 + 29056, tmp8)
        # tl.device_assert((0 <= tmp9) & (tmp9 < 29056), "index out of bounds: 0 <= tmp9 < 29056")
        tmp10 = tl.load(in_ptr1 + (tmp9 + (29056*r0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp11 = -tmp10
        tmp12 = 0.0
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp2, rmask)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp17 = tl.load(out_ptr1 + (r0), rmask, eviction_policy='evict_first')
        tmp18 = tl.load(in_ptr0 + (1 + (512*(r0 // 511)) + (r0 % 511)), rmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.full([1, 1], 0, tl.int64)
        tmp20 = tl.where(tmp17, tmp18, tmp19)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp20, rmask)
    tmp21 = tmp5.to(tl.float32)
    tmp22 = tmp15 / tmp21
    tl.store(out_ptr3 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp22, None)
''')
