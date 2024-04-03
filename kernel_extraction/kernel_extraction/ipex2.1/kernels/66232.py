

# Original file: ./hf_T5_generate__41_inference_81.21/hf_T5_generate__41_inference_81.21_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/qt/cqtpl26aslsmkr6zshnhanlcie6tjpj6ac53tz4j5oqxxli3da2z.py
# Source Nodes: [add_26, add_28, add_31, add_34, add_35, l__self___decoder_block_2_layer_1_dropout, l__self___decoder_block_2_layer__1__dropout, l__self___decoder_block_3_layer_0_dropout, l__self___decoder_block_3_layer_1_dropout, mean_11, mul_26, mul_27, pow_12, rsqrt_11, to_26, to_27], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_26 => add_30
# add_28 => add_32
# add_31 => add_35
# add_34 => add_38
# add_35 => add_39
# l__self___decoder_block_2_layer_1_dropout => clone_16
# l__self___decoder_block_2_layer__1__dropout => clone_18
# l__self___decoder_block_3_layer_0_dropout => clone_20
# l__self___decoder_block_3_layer_1_dropout => clone_22
# mean_11 => mean_11
# mul_26 => mul_26
# mul_27 => mul_27
# pow_12 => pow_12
# rsqrt_11 => rsqrt_11
# to_26 => convert_element_type_43
# to_27 => convert_element_type_44
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 512.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-06
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tmp21 = tmp9 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp15 * tmp22
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp8, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp23, rmask)
''')
