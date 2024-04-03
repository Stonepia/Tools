

# Original file: ./hf_T5_generate__71_inference_111.51/hf_T5_generate__71_inference_111.51.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/amp_fp16/ln/clnajk5nyub6ipw4qylnwwezoumzgq5xqr2qt62f2f26f4v7wcaa.py
# Source Nodes: [add_10, add_11, add_6, l__self___decoder_block_0_layer__1__dense_relu_dense_wi, l__self___decoder_embed_tokens, mean_2, mul_8, mul_9, pow_3, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_10 => add_14
# add_11 => add_15
# add_6 => add_9
# l__self___decoder_block_0_layer__1__dense_relu_dense_wi => convert_element_type_20
# l__self___decoder_embed_tokens => embedding
# mean_2 => mean_2
# mul_8 => mul_8
# mul_9 => mul_9
# pow_3 => pow_3
# rsqrt_2 => rsqrt_2
triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9 = async_compile.triton('triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_mean_mul_pow_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp7 = tl.load(in_ptr3 + (r0), rmask, other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r0), rmask, other=0.0)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 32128, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 32128), "index out of bounds: 0 <= tmp2 < 32128")
    tmp3 = tl.load(in_ptr1 + (r0 + (512*tmp2)), rmask, other=0.0)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
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
    tmp22 = tmp15 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp23, rmask)
''')
