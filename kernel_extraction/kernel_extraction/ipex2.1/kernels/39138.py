

# Original file: ./DebertaV2ForMaskedLM__0_backward_207.1/DebertaV2ForMaskedLM__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/7t/c7tlgbevn5hzyotou4mnbrio2os4xuitvfu6csqcjf2ke5lndw65.py
# Source Nodes: [iadd, l__mod___deberta_embeddings_layer_norm, trampoline_autograd_apply], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# iadd => add
# l__mod___deberta_embeddings_layer_norm => convert_element_type
# trampoline_autograd_apply => full_default_1
triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_21 = async_compile.triton('triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_21', '''
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*bf16', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_21', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, out_ptr0, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp1 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr4 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = 0.0
        tmp9 = tl.where(tmp0, tmp8, tmp7)
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp12 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
        tl.store(out_ptr0 + (r1 + (1536*x0)), tmp12, rmask & xmask)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    x2 = xindex % 512
    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    _tmp33 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr6 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr7 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp19 * tmp21
        tmp25 = tmp23 + tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp26 - tmp27
        tmp30 = tmp28 * tmp29
        tmp31 = tmp22 * tmp30
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = _tmp33 + tmp32
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp33 = tl.sum(_tmp33, 1)[:, None]
    tmp53 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp37 = tl.load(out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp38 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr6 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp44 = tl.load(in_ptr7 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = 1536.0
        tmp36 = tmp29 / tmp35
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp37 * tmp39
        tmp41 = tmp40 * tmp35
        tmp42 = tmp41 - tmp17
        tmp45 = tmp43 + tmp44
        tmp46 = tmp45.to(tl.float32)
        tmp47 = tmp46 - tmp27
        tmp48 = tmp47 * tmp29
        tmp49 = tmp48 * tmp33
        tmp50 = tmp42 - tmp49
        tmp51 = tmp36 * tmp50
        tmp52 = tmp51.to(tl.float32)
        tmp54 = tl.where(tmp53 < 0, tmp53 + 128100, tmp53)
        tmp55 = tl.full([1, 1], 0, tl.int64)
        tmp56 = tmp53 == tmp55
        tmp57 = tmp52.to(tl.float32)
        tmp58 = 0.0
        tmp59 = tl.where(tmp56, tmp58, tmp57)
        tl.store(out_ptr3 + (r1 + (1536*x0)), tmp52, rmask & xmask)
        tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (1536*tmp54), [XBLOCK, RBLOCK])), tmp59, rmask & xmask)
''')
