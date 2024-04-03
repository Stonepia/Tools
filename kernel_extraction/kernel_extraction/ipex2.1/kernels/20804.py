

# Original file: ./AlbertForMaskedLM__0_backward_135.1/AlbertForMaskedLM__0_backward_135.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/xq/cxqk5c355tqw77c6w252fbxawikocozu646ivqarrxj6e24hdblg.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.add, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => full_default_2
triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', '''
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
    size_hints=[2048, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_add_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 30000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (30000*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp20 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp22 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr4 + (r1 + (30000*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr0 + (r1 + (30000*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr5 + (r1 + (30000*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.full([1, 1], -100, tl.int64)
        tmp19 = tmp1 != tmp18
        tmp24 = tmp21 / tmp23
        tmp25 = 0.0
        tmp26 = tl.where(tmp19, tmp24, tmp25)
        tmp27 = tmp17 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tl.exp(tmp30)
        tmp32 = tmp31 * tmp14
        tmp33 = tmp28 - tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp16 + tmp34
        tl.store(out_ptr1 + (r1 + (30000*x0)), tmp35, rmask)
''')
