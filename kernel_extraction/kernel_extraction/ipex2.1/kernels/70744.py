

# Original file: ./DistillGPT2__0_forward_97.0/DistillGPT2__0_forward_97.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/3f/c3fbsuzbdwqjvxmzdxsonvj52qph53jq3yhlnzvk6rx3x7hxpmzh.py
# Source Nodes: [full, full_1, l__mod___transformer_h_0_attn_attn_dropout, softmax, truediv, where], Original ATen: [aten._softmax, aten.div, aten.full, aten.native_dropout, aten.where]
# full => full_default
# full_1 => full_default_1
# l__mod___transformer_h_0_attn_attn_dropout => gt_1, mul_4, mul_5
# softmax => amax, convert_element_type_2, convert_element_type_3, div_1, exp, sub_1, sum_1
# truediv => div
# where => where
triton_red_fused__softmax_div_full_native_dropout_where_4 = async_compile.triton('triton_red_fused__softmax_div_full_native_dropout_where_4', '''
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
    size_hints=[131072, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_div_full_native_dropout_where_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_div_full_native_dropout_where_4(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = 8.0
        tmp3 = tmp1 / tmp2
        tmp4 = -65504.0
        tmp5 = tl.where(tmp0, tmp3, tmp4)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
        tmp11 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = 8.0
        tmp13 = tmp11 / tmp12
        tmp14 = -65504.0
        tmp15 = tl.where(tmp10, tmp13, tmp14)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp16 - tmp8
        tmp18 = tl.exp(tmp17)
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask, tmp21, _tmp20)
        tmp22 = tl.load(in_ptr2 + load_seed_offset)
        tmp23 = r2 + (512*x3)
        tmp24 = tl.rand(tmp22, (tmp23).to(tl.uint32))
        tl.store(out_ptr2 + (r2 + (512*x3)), tmp24, rmask)
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp25 = tl.load(out_ptr2 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr0 + (r2 + (1024*x0)), rmask, eviction_policy='evict_last')
        tmp30 = tl.load(in_ptr1 + (r2 + (512*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 0.1
        tmp28 = tmp26 > tmp27
        tmp31 = 8.0
        tmp32 = tmp30 / tmp31
        tmp33 = -65504.0
        tmp34 = tl.where(tmp29, tmp32, tmp33)
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 - tmp8
        tmp37 = tl.exp(tmp36)
        tmp38 = tmp37 / tmp20
        tmp39 = tmp38.to(tl.float32)
        tmp40 = tmp28.to(tl.float32)
        tmp41 = tmp40 * tmp39
        tmp42 = 1.1111111111111112
        tmp43 = tmp41 * tmp42
        tl.store(out_ptr3 + (r2 + (512*x3)), tmp28, rmask)
        tl.store(out_ptr4 + (r2 + (512*x3)), tmp39, rmask)
        tl.store(out_ptr5 + (r2 + (512*x3)), tmp43, rmask)
''')
