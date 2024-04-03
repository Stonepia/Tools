

# Original file: ./hf_T5___60.0/hf_T5___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_bf16/2b/c2bjaiouflllleh2bitsajzx6eicfrwrnmn75q3fikxmtvurt77k.py
# Source Nodes: [add_10, add_12, add_14, add_16, add_17, l__self___model_encoder_block_1_layer__1__dropout, l__self___model_encoder_block_2_layer_0_dropout, l__self___model_encoder_block_2_layer__1__dropout, l__self___model_encoder_block_3_layer_0_dropout, l__self___model_encoder_block_3_layer__1__dense_relu_dense_wi, mean_7, mul_17, mul_18, pow_8, rsqrt_7], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_10 => add_13
# add_12 => add_16
# add_14 => add_18
# add_16 => add_21
# add_17 => add_22
# l__self___model_encoder_block_1_layer__1__dropout => clone_10
# l__self___model_encoder_block_2_layer_0_dropout => clone_13
# l__self___model_encoder_block_2_layer__1__dropout => clone_15
# l__self___model_encoder_block_3_layer_0_dropout => clone_18
# l__self___model_encoder_block_3_layer__1__dense_relu_dense_wi => convert_element_type_58
# mean_7 => mean_7
# mul_17 => mul_17
# mul_18 => mul_18
# pow_8 => pow_8
# rsqrt_7 => rsqrt_7
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13', '''
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
    size_hints=[2048, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_out_ptr0 + (r1 + (512*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + (512*x0)), rmask, other=0.0).to(tl.float32)
    tmp18 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp19 = 512.0
    tmp20 = tmp17 / tmp19
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp12 * tmp23
    tmp25 = tmp18 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tl.store(in_out_ptr0 + (r1 + (512*x0)), tmp12, rmask)
    tl.store(out_ptr1 + (r1 + (512*x0)), tmp26, rmask)
''')
