

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/7t/c7t63wswnry7cukoo6bo4zh5lybqskqlpfs47r4xw2i3zastshjd.py
# Source Nodes: [add_4, any_1, isinf, l__self___block_0_layer_0_dropout, l__self___embed_tokens], Original ATen: [aten.add, aten.any, aten.clone, aten.embedding, aten.isinf]
# add_4 => add_6
# any_1 => any_1
# isinf => isinf
# l__self___block_0_layer_0_dropout => clone_3
# l__self___embed_tokens => embedding
triton_red_fused_add_any_clone_embedding_isinf_4 = async_compile.triton('triton_red_fused_add_any_clone_embedding_isinf_4', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[128, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_any_clone_embedding_isinf_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_any_clone_embedding_isinf_4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + ((16*x0) + (r1 // 512)), xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r1 + (8192*x0)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + ((512*tmp1) + (r1 % 512)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = libdevice.isinf(tmp4).to(tl.int1)
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 | tmp6
        _tmp7 = tl.where(xmask, tmp8, _tmp7)
    tmp7 = triton_helpers.any(_tmp7.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''')
