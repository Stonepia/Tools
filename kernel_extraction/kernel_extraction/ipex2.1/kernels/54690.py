

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/zz/czzxktqgtlruzwmcpggwwphloxrc7nimhrjhrzujdgkw7hvzzu3d.py
# Source Nodes: [l__mod___deberta_encoder_layer_23_attention_output_layer_norm, trampoline_autograd_apply], Original ATen: [aten.add, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___deberta_encoder_layer_23_attention_output_layer_norm => convert_element_type_405
# trampoline_autograd_apply => full_default_1
triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_11', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_masked_fill_mul_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 - tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tmp6 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr6 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp20 = 1536.0
        tmp21 = tmp14 / tmp20
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp29 = tmp28 * tmp20
        tmp30 = tmp29 - tmp8
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32 - tmp12
        tmp34 = tmp33 * tmp14
        tmp35 = tmp34 * tmp18
        tmp36 = tmp30 - tmp35
        tmp37 = tmp21 * tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp40 = 0.0
        tmp41 = tl.where(tmp39, tmp40, tmp38)
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp38, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp43, rmask & xmask)
''')
