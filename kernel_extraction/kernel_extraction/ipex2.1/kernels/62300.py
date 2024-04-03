

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/tt/cttlmcbftnjqhingc7aeojykl6b62uyeruzsgismca4el33ifncy.py
# Source Nodes: [add_6, add_7, l__mod___encoder_block_0_layer__1__dropout, l__mod___encoder_block_1_layer_0_self_attention_q, mean_2, mul_7, mul_8, pow_3, rsqrt_2, to_7, to_8], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add_6 => add_8
# add_7 => add_9
# l__mod___encoder_block_0_layer__1__dropout => gt_5, mul_15, mul_16
# l__mod___encoder_block_1_layer_0_self_attention_q => view_26
# mean_2 => mean_2
# mul_7 => mul_17
# mul_8 => mul_18
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
# to_7 => convert_element_type_10
# to_8 => convert_element_type_11
triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_8 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_8', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*bf16', 4: '*bf16', 5: '*i1', 6: '*bf16', 7: 'i32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, load_seed_offset, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp6 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp24 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp7 = tmp5.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = 1.1111111111111112
    tmp11 = tmp9 * tmp10
    tmp12 = tmp6 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 512.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp25 = tmp13 * tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp12, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask)
''')