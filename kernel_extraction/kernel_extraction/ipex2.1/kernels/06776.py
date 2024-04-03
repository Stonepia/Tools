

# Original file: ./DistilBertForQuestionAnswering__0_forward_97.0/DistilBertForQuestionAnswering__0_forward_97.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/z7/cz7v7noyt4hq2jrywimocilwulaqevkmdz2aq7kxdzoy2nfk2ikw.py
# Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_dropout, masked_fill, softmax, tensor], Original ATen: [aten._softmax, aten.lift_fresh, aten.masked_fill, aten.native_dropout]
# l__mod___distilbert_transformer_layer_0_attention_dropout => gt_1, mul_4, mul_5
# masked_fill => where
# softmax => amax, div_1, exp, sub_1, sum_1
# tensor => full_default
triton_red_fused__softmax_lift_fresh_masked_fill_native_dropout_5 = async_compile.triton('triton_red_fused__softmax_lift_fresh_masked_fill_native_dropout_5', '''
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
    size_hints=[524288, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_lift_fresh_masked_fill_native_dropout_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_lift_fresh_masked_fill_native_dropout_5(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 393216
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x3 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = -3.4028234663852886e+38
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = triton_helpers.maximum(_tmp5, tmp4)
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
    tmp5 = triton_helpers.max2(_tmp5, 1)[:, None]
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp7 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp8 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = -3.4028234663852886e+38
        tmp10 = tl.where(tmp7, tmp9, tmp8)
        tmp11 = tmp10 - tmp5
        tmp12 = tl.exp(tmp11)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
        tmp16 = tl.load(in_ptr2 + load_seed_offset)
        tmp17 = r2 + (128*x3)
        tmp18 = tl.rand(tmp16, (tmp17).to(tl.uint32))
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp18, rmask)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp19 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp23 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp20 = 0.1
        tmp21 = tmp19 > tmp20
        tmp24 = -3.4028234663852886e+38
        tmp25 = tl.where(tmp22, tmp24, tmp23)
        tmp26 = tmp25 - tmp5
        tmp27 = tl.exp(tmp26)
        tmp28 = tmp27 / tmp14
        tmp29 = tmp21.to(tl.float32)
        tmp30 = tmp29 * tmp28
        tmp31 = 1.1111111111111112
        tmp32 = tmp30 * tmp31
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp21, rmask)
        tl.store(out_ptr4 + (r2 + (128*x3)), tmp28, rmask)
        tl.store(out_ptr5 + (r2 + (128*x3)), tmp32, rmask)
''')
