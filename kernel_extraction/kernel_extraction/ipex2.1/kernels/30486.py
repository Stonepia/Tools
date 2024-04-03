

# Original file: ./hf_T5_generate__63_inference_103.43/hf_T5_generate__63_inference_103.43_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/qj/cqjflv3mgskspkjdz5ujuantztqj6vqjwluaxxyplaf5usglgbi2.py
# Source Nodes: [add_10, add_11, add_6, l__self___decoder_block_0_layer_0_dropout, l__self___decoder_block_0_layer_1_dropout, l__self___decoder_embed_tokens, mean_2, mul_8, mul_9, pow_3, rsqrt_2, to_8, to_9], Original ATen: [aten._to_copy, aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_10 => add_14
# add_11 => add_15
# add_6 => add_9
# l__self___decoder_block_0_layer_0_dropout => clone_2
# l__self___decoder_block_0_layer_1_dropout => clone_4
# l__self___decoder_embed_tokens => embedding
# mean_2 => mean_2
# mul_8 => mul_8
# mul_9 => mul_9
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
# to_8 => convert_element_type_13
# to_9 => convert_element_type_14
triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9', '''
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
    meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_clone_embedding_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r0), rmask, other=0.0).to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 32128, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 32128), "index out of bounds: 0 <= tmp2 < 32128")
    tmp3 = tl.load(in_ptr1 + (r0 + (512*tmp2)), rmask, other=0.0).to(tl.float32)
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp15 = 512.0
    tmp16 = tmp13 / tmp15
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp8 * tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp14 * tmp21
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp22, rmask)
''')
