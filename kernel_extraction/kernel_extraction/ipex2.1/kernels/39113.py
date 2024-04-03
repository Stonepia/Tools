

# Original file: ./DebertaV2ForMaskedLM__0_backward_207.1/DebertaV2ForMaskedLM__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/uj/cuj2xpnqxbtlm2zl4dchjbb2kij5xmsfgbcoc7nuxlu5zhjw3wvm.py
# Source Nodes: [iadd, l__self___deberta_embeddings_layer_norm, trampoline_autograd_apply], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm, aten.native_layer_norm_backward]
# iadd => add
# l__self___deberta_embeddings_layer_norm => mul, sub
# trampoline_autograd_apply => full_default_1
triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_26 = async_compile.triton('triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_26', '''
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
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*i64', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_26', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_native_layer_norm_backward_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp1 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 + tmp6
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 + tmp9
        tmp11 = 0.0
        tmp12 = tl.where(tmp0, tmp11, tmp10)
        tmp13 = 1.1111111111111112
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
        tl.store(in_out_ptr0 + (r1 + (1536*x0)), tmp14, rmask & xmask)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    x2 = xindex % 512
    tmp26 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    _tmp32 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp24 = tl.load(in_ptr6 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp25 = tmp23 + tmp24
        tmp27 = tmp25 - tmp26
        tmp29 = tmp27 * tmp28
        tmp30 = tmp22 * tmp29
        tmp31 = tl.broadcast_to(tmp30, [XBLOCK, RBLOCK])
        tmp33 = _tmp32 + tmp31
        _tmp32 = tl.where(rmask & xmask, tmp33, _tmp32)
    tmp32 = tl.sum(_tmp32, 1)[:, None]
    tmp49 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp36 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp37 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp41 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp42 = tl.load(in_ptr6 + (r1 + (1536*x2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp34 = 1536.0
        tmp35 = tmp28 / tmp34
        tmp38 = tmp36 * tmp37
        tmp39 = tmp38 * tmp34
        tmp40 = tmp39 - tmp18
        tmp43 = tmp41 + tmp42
        tmp44 = tmp43 - tmp26
        tmp45 = tmp44 * tmp28
        tmp46 = tmp45 * tmp32
        tmp47 = tmp40 - tmp46
        tmp48 = tmp35 * tmp47
        tmp50 = tl.where(tmp49 < 0, tmp49 + 128100, tmp49)
        tmp51 = tl.full([1, 1], 0, tl.int64)
        tmp52 = tmp49 == tmp51
        tmp53 = 0.0
        tmp54 = tl.where(tmp52, tmp53, tmp48)
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp48, rmask & xmask)
        tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (1536*tmp50), [XBLOCK, RBLOCK])), tmp54, rmask & xmask)
''')
