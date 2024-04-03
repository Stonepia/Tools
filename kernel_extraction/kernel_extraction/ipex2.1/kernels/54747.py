

# Original file: ./DebertaV2ForQuestionAnswering__0_backward_207.1/DebertaV2ForQuestionAnswering__0_backward_207.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/fp/cfpewoc4igjntdt4egpkzb5gxhhaj5jhw2gsyjfsagrvwawiumd5.py
# Source Nodes: [trampoline_autograd_apply], Original ATen: [aten.add, aten.embedding_dense_backward, aten.masked_fill, aten.mul, aten.native_layer_norm_backward]
# trampoline_autograd_apply => full_default_1
triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_20 = async_compile.triton('triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_20', '''
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
    meta={'signature': {0: '*fp32', 1: '*i1', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i64', 9: '*i64', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr3', 'out_ptr4'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_20', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_embedding_dense_backward_masked_fill_mul_native_layer_norm_backward_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp20 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp19 = tmp17 * tmp18
        tmp21 = tmp19 * tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tmp25 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp26 = tl.load(in_out_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp32 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp28 = tmp26 * tmp27
        tmp29 = 1536.0
        tmp30 = tmp28 * tmp29
        tmp31 = tmp30 - tmp15
        tmp33 = tmp32 * tmp23
        tmp34 = tmp31 - tmp33
        tmp35 = tmp25 * tmp34
        tmp37 = tl.where(tmp36 < 0, tmp36 + 512, tmp36)
        tmp38 = tl.full([1, 1], -1, tl.int64)
        tmp39 = tmp36 == tmp38
        tmp40 = 0.0
        tmp41 = tl.where(tmp39, tmp40, tmp35)
        tmp43 = tl.where(tmp42 < 0, tmp42 + 128100, tmp42)
        tmp44 = tl.full([1, 1], 0, tl.int64)
        tmp45 = tmp42 == tmp44
        tmp46 = tl.where(tmp45, tmp40, tmp35)
        tl.atomic_add(out_ptr3 + (tl.broadcast_to(r1 + (1536*tmp37), [XBLOCK, RBLOCK])), tmp41, rmask & xmask)
        tl.atomic_add(out_ptr4 + (tl.broadcast_to(r1 + (1536*tmp43), [XBLOCK, RBLOCK])), tmp46, rmask & xmask)
''')
