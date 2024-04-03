

# Original file: ./DebertaV2ForMaskedLM__0_backward_207.1/DebertaV2ForMaskedLM__0_backward_207.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/yv/cyv7f4oswldo3x65og723avxazmmrill4qokvpqktodg6pr37mqs.py
# Source Nodes: [iadd, l__mod___deberta_embeddings_layer_norm, trampoline_autograd_apply], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# iadd => add
# l__mod___deberta_embeddings_layer_norm => mul, sub
# trampoline_autograd_apply => full_default_1
triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_20 = async_compile.triton('triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_20', '''
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
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp1 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = 0.0
        tmp9 = tl.where(tmp0, tmp8, tmp7)
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
        tl.store(in_out_ptr0 + (r1 + (1536*x0)), tmp11, rmask & xmask)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    x2 = xindex % 512
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr6 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp22 = tmp20 + tmp21
        tmp24 = tmp22 - tmp23
        tmp26 = tmp24 * tmp25
        tmp27 = tmp19 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask & xmask, tmp30, _tmp29)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tmp46 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp33 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp34 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp38 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp39 = tl.load(in_ptr6 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = 1536.0
        tmp32 = tmp25 / tmp31
        tmp35 = tmp33 * tmp34
        tmp36 = tmp35 * tmp31
        tmp37 = tmp36 - tmp15
        tmp40 = tmp38 + tmp39
        tmp41 = tmp40 - tmp23
        tmp42 = tmp41 * tmp25
        tmp43 = tmp42 * tmp29
        tmp44 = tmp37 - tmp43
        tmp45 = tmp32 * tmp44
        tmp47 = tl.where(tmp46 < 0, tmp46 + 128100, tmp46)
        tmp48 = tl.full([1, 1], 0, tl.int64)
        tmp49 = tmp46 == tmp48
        tmp50 = 0.0
        tmp51 = tl.where(tmp49, tmp50, tmp45)
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp45, rmask & xmask)
        tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (1536*tmp47), [XBLOCK, RBLOCK])), tmp51, rmask & xmask)
''')
