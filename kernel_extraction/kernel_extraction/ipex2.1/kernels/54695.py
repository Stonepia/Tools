

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/56/c56hevtvu4cijs73hwkxdgbxe3ios2hfvgjdfm7ujnd7ygyx5m6v.py
# Source Nodes: [l__mod___deberta_encoder_layer_22_output_layer_norm, trampoline_autograd_apply], Original ATen: [aten.add, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___deberta_encoder_layer_22_output_layer_norm => convert_element_type_394
# trampoline_autograd_apply => full_default_1
triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_16 = async_compile.triton('triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_16', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: '*fp32', 10: '*bf16', 11: '*bf16', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
        tl.store(out_ptr0 + (r1 + (1536*x0)), tmp10, rmask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 - tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tmp14 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp27 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp38 = tl.load(in_ptr8 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp25 = 1536.0
        tmp26 = tmp19 / tmp25
        tmp28 = tmp27 * tmp25
        tmp29 = tmp28 - tmp12
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 - tmp17
        tmp33 = tmp32 * tmp19
        tmp34 = tmp33 * tmp23
        tmp35 = tmp29 - tmp34
        tmp36 = tmp26 * tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp39 = 0.0
        tmp40 = tl.where(tmp38, tmp39, tmp37)
        tmp41 = 1.1111111111111112
        tmp42 = tmp40 * tmp41
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp37, rmask & xmask)
        tl.store(out_ptr4 + (r1 + (1536*x0)), tmp42, rmask & xmask)
''')
