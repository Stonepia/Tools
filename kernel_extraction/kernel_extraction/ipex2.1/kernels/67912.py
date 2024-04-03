

# Original file: ./hf_T5_generate__49_inference_89.29/hf_T5_generate__49_inference_89.29_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/l6/cl6awx5aqgm46xwvo3gn7n6euuc2qpprme633jngt6lbvcyi6cgq.py
# Source Nodes: [add_6, add_7, l__self___decoder_block_0_layer_0_dropout, l__self___decoder_embed_tokens, mean_1, mul_6, mul_7, pow_2, rsqrt_1], Original ATen: [aten.add, aten.clone, aten.embedding, aten.mean, aten.mul, aten.pow, aten.rsqrt]
# add_6 => add_9
# add_7 => add_10
# l__self___decoder_block_0_layer_0_dropout => clone_2
# l__self___decoder_embed_tokens => embedding
# mean_1 => mean_1
# mul_6 => mul_6
# mul_7 => mul_7
# pow_2 => pow_2
# rsqrt_1 => rsqrt_1
triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_7 = async_compile.triton('triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_7', '''
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
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_clone_embedding_mean_mul_pow_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel):
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
    tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0.0)
    tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0.0)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 32128, tmp1)
    # tl.device_assert((0 <= tmp2) & (tmp2 < 32128), "index out of bounds: 0 <= tmp2 < 32128")
    tmp3 = tl.load(in_ptr1 + (r0 + (512*tmp2)), rmask, other=0.0)
    tmp5 = tmp3 + tmp4
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 512.0
    tmp13 = tmp10 / tmp12
    tmp14 = 1e-06
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp5 * tmp16
    tmp18 = tmp11 * tmp17
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp18, rmask)
''')
