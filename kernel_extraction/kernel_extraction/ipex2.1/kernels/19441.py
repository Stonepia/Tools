

# Original file: ./DistilBertForMaskedLM__0_forward_97.0/DistilBertForMaskedLM__0_forward_97.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/hc/chcupoja3q7hwyrvrwogzmbabyqm2ychstdd3hpylxfuo2aboxx5.py
# Source Nodes: [l__mod___distilbert_transformer_layer_0_attention_dropout, masked_fill, softmax], Original ATen: [aten._softmax, aten.masked_fill, aten.native_dropout]
# l__mod___distilbert_transformer_layer_0_attention_dropout => gt_1, mul_4, mul_5
# masked_fill => full_default, where
# softmax => amax, convert_element_type_3, convert_element_type_4, div_1, exp, sub_1, sum_1
triton_red_fused__softmax_masked_fill_native_dropout_5 = async_compile.triton('triton_red_fused__softmax_masked_fill_native_dropout_5', '''
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
    size_hints=[262144, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_masked_fill_native_dropout_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_masked_fill_native_dropout_5(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 196608
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 1536)
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = -65504.0
        tmp3 = tl.where(tmp0, tmp2, tmp1)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = triton_helpers.maximum(_tmp6, tmp5)
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = triton_helpers.max2(_tmp6, 1)[:, None]
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = -65504.0
        tmp11 = tl.where(tmp8, tmp10, tmp9)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp12 - tmp6
        tmp14 = tl.exp(tmp13)
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask, tmp17, _tmp16)
        tmp18 = tl.load(in_ptr2 + load_seed_offset)
        tmp19 = r2 + (128*x3)
        tmp20 = tl.rand(tmp18, (tmp19).to(tl.uint32))
        tl.store(out_ptr2 + (r2 + (128*x3)), tmp20, rmask)
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(out_ptr2 + (r2 + (128*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp25 = tl.load(in_ptr0 + (r2 + (128*x1)), rmask, eviction_policy='evict_last')
        tmp26 = tl.load(in_ptr1 + (r2 + (128*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = 0.1
        tmp24 = tmp22 > tmp23
        tmp27 = -65504.0
        tmp28 = tl.where(tmp25, tmp27, tmp26)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp6
        tmp31 = tl.exp(tmp30)
        tmp32 = tmp31 / tmp16
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp24.to(tl.float32)
        tmp35 = tmp34 * tmp33
        tmp36 = 1.1111111111111112
        tmp37 = tmp35 * tmp36
        tl.store(out_ptr3 + (r2 + (128*x3)), tmp24, rmask)
        tl.store(out_ptr4 + (r2 + (128*x3)), tmp33, rmask)
        tl.store(out_ptr5 + (r2 + (128*x3)), tmp37, rmask)
''')
