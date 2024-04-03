

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/x7/cx7u7rzzto3hlpx3uq63z4sgq6gzd7nzezuwc7t3vfsgtecszpys.py
# Source Nodes: [trampoline_autograd_apply], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
# trampoline_autograd_apply => full_default_1
triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_23 = async_compile.triton('triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_23', '''
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
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_23', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp20 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 * tmp23
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp29 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp35 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tmp29 * tmp30
        tmp32 = 1536.0
        tmp33 = tmp31 * tmp32
        tmp34 = tmp33 - tmp18
        tmp36 = tmp35 * tmp26
        tmp37 = tmp34 - tmp36
        tmp38 = tmp28 * tmp37
        tmp40 = tl.where(tmp39 < 0, tmp39 + 512, tmp39)
        tmp41 = tl.full([1, 1], -1, tl.int64)
        tmp42 = tmp39 == tmp41
        tmp43 = 0.0
        tmp44 = tl.where(tmp42, tmp43, tmp38)
        tmp46 = tl.where(tmp45 < 0, tmp45 + 128100, tmp45)
        tmp47 = tl.full([1, 1], 0, tl.int64)
        tmp48 = tmp45 == tmp47
        tmp49 = tl.where(tmp48, tmp43, tmp38)
        tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (1536*tmp40), [XBLOCK, RBLOCK])), tmp44, rmask & xmask)
        tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (1536*tmp46), [XBLOCK, RBLOCK])), tmp49, rmask & xmask)
''')
