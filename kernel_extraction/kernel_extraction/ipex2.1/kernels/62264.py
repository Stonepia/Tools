

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/aq/caqemfjfuju32l4bdbhjamnvi32evmw7pkljcilnfd2wzhh5mywd.py
# Source Nodes: [add_26, add_27, l__self___decoder_block_0_layer_1_enc_dec_attention_k, l__self___encoder_block_5_layer__1__dropout, l__self___encoder_dropout_1, mean_12, mul_27, mul_28, pow_13, rsqrt_12], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.native_dropout, aten.pow, aten.rsqrt, aten.view]
# add_26 => add_33
# add_27 => add_34
# l__self___decoder_block_0_layer_1_enc_dec_attention_k => convert_element_type_109, view_176
# l__self___encoder_block_5_layer__1__dropout => gt_25, mul_75, mul_76
# l__self___encoder_dropout_1 => gt_26, mul_79, mul_80
# mean_12 => mean_12
# mul_27 => mul_77
# mul_28 => mul_78
# pow_13 => pow_13
# rsqrt_12 => rsqrt_12
triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_16 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_16', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*i1', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_native_dropout_pow_rsqrt_view_16(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr3, out_ptr4, out_ptr5, load_seed_offset, load_seed_offset1, xnumel, rnumel):
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
    tmp7 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp11 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = r1 + (512*x0)
    tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.1
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 1.1111111111111112
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10.to(tl.float32)
    tmp13 = tmp11 + tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 512.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tl.load(in_ptr0 + load_seed_offset1)
    tmp25 = tl.rand(tmp24, (tmp1).to(tl.uint32))
    tmp26 = tmp25 > tmp4
    tmp27 = tmp26.to(tl.float32)
    tmp29 = tmp13 * tmp23
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 * tmp30
    tmp32 = tmp31 * tmp9
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp5, rmask)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp10, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr3 + (r1 + (512*x0)), tmp26, rmask)
    tl.store(out_ptr4 + (r1 + (512*x0)), tmp32, rmask)
    tl.store(out_ptr5 + (r1 + (512*x0)), tmp33, rmask)
''')
