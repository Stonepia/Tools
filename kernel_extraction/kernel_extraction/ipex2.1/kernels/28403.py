

# Original file: ./XLNetLMHeadModel__0_forward_565.0/XLNetLMHeadModel__0_forward_565.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/3j/c3jnm2drtltjkv2tpb5nfly6o65u6jxujfts6aikntto5fuclasd.py
# Source Nodes: [add_2, add_3, index_select, l__self___transformer_layer_0_rel_attn_dropout, mul, softmax], Original ATen: [aten._softmax, aten.add, aten.index_select, aten.mul, aten.native_dropout]
# add_2 => add_2
# add_3 => add_3
# index_select => index
# l__self___transformer_layer_0_rel_attn_dropout => gt_2, mul_7, mul_8
# mul => mul_6
# softmax => amax, convert_element_type_13, convert_element_type_14, div_1, exp, sub, sum_1
triton_red_fused__softmax_add_index_select_mul_native_dropout_7 = async_compile.triton('triton_red_fused__softmax_add_index_select_mul_native_dropout_7', '''
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
    size_hints=[65536, 512],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*i64', 3: '*fp32', 4: '*i1', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_add_index_select_mul_native_dropout_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__softmax_add_index_select_mul_native_dropout_7(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = xindex % 512
    x1 = (xindex // 512) % 16
    x2 = (xindex // 8192)
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = 0.0
        tmp4 = tmp2 + tmp3
        tmp5 = 0.125
        tmp6 = tmp4 * tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp11 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = 0.0
        tmp15 = tmp13 + tmp14
        tmp16 = 0.125
        tmp17 = tmp15 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp24 = tl.load(in_ptr2 + load_seed_offset)
        tmp25 = r3 + (512*x4)
        tmp26 = tl.rand(tmp24, (tmp25).to(tl.uint32))
        tl.store(out_ptr2 + (r3 + (512*x4)), tmp26, rmask)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp27 = tl.load(out_ptr2 + (r3 + (512*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp31 = tl.load(in_ptr0 + (r3 + (512*x4)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr1 + (512 + r3 + (1023*x0) + (524288*x1) + (524288*((r3 + (1023*x0)) // 523776)) + (8388608*x2) + (8388608*((r3 + (1023*x0) + (523776*x1)) // 8380416))), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tmp27.to(tl.float32)
        tmp29 = 0.1
        tmp30 = tmp28 > tmp29
        tmp33 = tmp31 + tmp32
        tmp34 = 0.0
        tmp35 = tmp33 + tmp34
        tmp36 = 0.125
        tmp37 = tmp35 * tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp38 - tmp9
        tmp40 = tl.exp(tmp39)
        tmp41 = tmp40 / tmp22
        tmp42 = tmp41.to(tl.float32)
        tmp43 = tmp30.to(tl.float32)
        tmp44 = tmp43 * tmp42
        tmp45 = 1.1111111111111112
        tmp46 = tmp44 * tmp45
        tl.store(out_ptr3 + (r3 + (512*x4)), tmp30, rmask)
        tl.store(out_ptr4 + (r3 + (512*x4)), tmp42, rmask)
        tl.store(out_ptr5 + (r3 + (512*x4)), tmp46, rmask)
''')
