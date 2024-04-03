

# Original file: ./hf_T5___60.0/hf_T5___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/3l/c3lajydbbxfpodo253ezyu7rcdn3njskn5pkpx2y3vpec3gkgii5.py
# Source Nodes: [add_32, any_13, isinf_12, l__mod___model_decoder_block_0_layer_0_dropout, l__mod___model_encoder_embed_tokens_1], Original ATen: [aten.add, aten.any, aten.clone, aten.embedding, aten.isinf]
# add_32 => add_40
# any_13 => any_13
# isinf_12 => isinf_12
# l__mod___model_decoder_block_0_layer_0_dropout => clone_35
# l__mod___model_encoder_embed_tokens_1 => embedding_2
triton_per_fused_add_any_clone_embedding_isinf_5 = async_compile.triton('triton_per_fused_add_any_clone_embedding_isinf_5', '''
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
    size_hints=[1, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*i1', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_any_clone_embedding_isinf_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_add_any_clone_embedding_isinf_5(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.any(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''')