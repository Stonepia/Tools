

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/6m/c6my2jrr4y7idyt74whtct3u33s6jbkibc2wpmvg67ohu5vqeq4h.py
# Source Nodes: [trampoline_autograd_apply_1], Original ATen: [aten._to_copy, aten.add, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
# trampoline_autograd_apply_1 => full_default_5
triton_red_fused__to_copy_add_masked_fill_mul_native_layer_norm_backward_19 = async_compile.triton('triton_red_fused__to_copy_add_masked_fill_mul_native_layer_norm_backward_19', '''
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
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp32', 10: '*bf16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_masked_fill_mul_native_layer_norm_backward_19', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_masked_fill_mul_native_layer_norm_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask & xmask, tmp14, _tmp13)
        tl.store(out_ptr0 + (r1 + (1536*x0)), tmp11, rmask & xmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp15 * tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(rmask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp26 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr7 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp23 = 1536.0
        tmp24 = tmp22 * tmp23
        tmp25 = tmp24 - tmp13
        tmp27 = tmp26 * tmp19
        tmp28 = tmp25 - tmp27
        tmp29 = tmp21 * tmp28
        tmp31 = tmp29.to(tl.float32)
        tmp32 = 0.0
        tmp33 = tl.where(tmp30, tmp32, tmp31)
        tmp34 = 1.1111111111111112
        tmp35 = tmp33 * tmp34
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp29, rmask & xmask)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp35, rmask & xmask)
''')
