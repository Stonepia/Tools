

# Original file: ./hf_T5_generate__83_inference_123.63/hf_T5_generate__83_inference_123.63_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/ao/caowjyn6bsulmvdmfj3x3jb265exgeir3yagottxuiqnq2jcrj5k.py
# Source Nodes: [add_10, add_12, add_13, add_6, l__self___decoder_block_0_layer_0_dropout, l__self___decoder_block_0_layer_1_dropout, l__self___decoder_block_0_layer__1__dropout, l__self___decoder_embed_tokens, mean_3, mul_10, mul_11, pow_4, rsqrt_3, to_10, to_11], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_10 => add_14
# add_12 => add_16
# add_13 => add_17
# add_6 => add_9
# l__self___decoder_block_0_layer_0_dropout => clone_2
# l__self___decoder_block_0_layer_1_dropout => clone_4
# l__self___decoder_block_0_layer__1__dropout => clone_6
# l__self___decoder_embed_tokens => embedding
# mean_3 => mean_3
# mul_10 => mul_10
# mul_11 => mul_11
# pow_4 => pow_4
# rsqrt_3 => rsqrt_3
# to_10 => convert_element_type_15
# to_11 => convert_element_type_16
triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_11 = async_compile.triton('triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_11', '''
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
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp4 = tl.load(in_out_ptr0 + (r0), rmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 32128, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 32128), "index out of bounds: 0 <= tmp2 < 32128")
    tmp3 = tl.load(in_ptr1 + (r0 + (512*tmp2)), rmask, other=0.0).to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp17 = 512.0
    tmp18 = tmp15 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp10 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp16 * tmp23
    tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp9, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp24, rmask)
''')
