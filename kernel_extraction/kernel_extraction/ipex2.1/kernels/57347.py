

# Original file: ./BERT_pytorch___60.0/BERT_pytorch___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/hf/chf3gegj3spj6adprxuwavsgvfwkqzrtw3qyaxjsn3ltck2ueci5.py
# Source Nodes: [eq, masked_fill, softmax, truediv_1], Original ATen: [aten._softmax, aten.div, aten.eq, aten.masked_fill]
# eq => eq
# masked_fill => full_default, where
# softmax => amax, convert_element_type_10, convert_element_type_9, div_2, exp, sub_1, sum_1
# truediv_1 => div_1
triton_red_fused__softmax_div_eq_masked_fill_4 = async_compile.triton('triton_red_fused__softmax_div_eq_masked_fill_4', '''
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
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_div_eq_masked_fill_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_div_eq_masked_fill_4(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x2 = (xindex // 1536)
    x4 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.int64)
        tmp2 = tl.full([1, 1], 0, tl.int64)
        tmp3 = tmp1 == tmp2
        tmp5 = 8.0
        tmp6 = tmp4 / tmp5
        tmp7 = -998244352.0
        tmp8 = tl.where(tmp3, tmp7, tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp13 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last')
        tmp17 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp14 = tmp13.to(tl.int64)
        tmp15 = tl.full([1, 1], 0, tl.int64)
        tmp16 = tmp14 == tmp15
        tmp18 = 8.0
        tmp19 = tmp17 / tmp18
        tmp20 = -998244352.0
        tmp21 = tl.where(tmp16, tmp20, tmp19)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp11
        tmp24 = tl.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp28 = tl.load(in_ptr0 + (r3 + (128*x0) + (16384*x2)), rmask, eviction_policy='evict_last')
        tmp32 = tl.load(in_ptr1 + (r3 + (128*x4)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tmp28.to(tl.int64)
        tmp30 = tl.full([1, 1], 0, tl.int64)
        tmp31 = tmp29 == tmp30
        tmp33 = 8.0
        tmp34 = tmp32 / tmp33
        tmp35 = -998244352.0
        tmp36 = tl.where(tmp31, tmp35, tmp34)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp37 - tmp11
        tmp39 = tl.exp(tmp38)
        tmp40 = tmp39 / tmp26
        tmp41 = tmp40.to(tl.float32)
        tl.store(out_ptr2 + (r3 + (128*x4)), tmp41, rmask)
''')
