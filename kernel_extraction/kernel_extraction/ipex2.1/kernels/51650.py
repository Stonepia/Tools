

# Original file: ./hf_T5_generate__53_inference_93.33/hf_T5_generate__53_inference_93.33_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/aw/cawis62owjpp5ydmtgi2f7jyuypxepaf6c5dohguwnhbcf5uypol.py
# Source Nodes: [add_15, add_18, add_20, add_21, l__self___decoder_block_1_layer_0_dropout, l__self___decoder_block_1_layer_1_dropout, l__self___decoder_block_1_layer__1__dropout, mean_6, mul_16, mul_17, pow_7, rsqrt_6, to_16, to_17], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_15 => add_19
# add_18 => add_22
# add_20 => add_24
# add_21 => add_25
# l__self___decoder_block_1_layer_0_dropout => clone_8
# l__self___decoder_block_1_layer_1_dropout => clone_10
# l__self___decoder_block_1_layer__1__dropout => clone_12
# mean_6 => mean_6
# mul_16 => mul_16
# mul_17 => mul_17
# pow_7 => pow_7
# rsqrt_6 => rsqrt_6
# to_16 => convert_element_type_25
# to_17 => convert_element_type_26
triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14 = async_compile.triton('triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_mean_mul_pow_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp14 = 512.0
    tmp15 = tmp12 / tmp14
    tmp16 = 1e-06
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp7 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp13 * tmp20
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp21, rmask)
''')
