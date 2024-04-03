

# Original file: ./hf_T5_generate___60.0/hf_T5_generate___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/ym/cym6g7nresnhndgebywdavtnm5qgv4zsffnldgr2obphyesmqkjr.py
# Source Nodes: [add_4, add_6, any_2, clamp, isinf_1, l__self___block_0_layer_0_dropout, l__self___block_0_layer__1__dropout, l__self___embed_tokens, neg, where_1], Original ATen: [aten.add, aten.any, aten.clamp, aten.clone, aten.embedding, aten.isinf, aten.neg, aten.scalar_tensor, aten.where]
# add_4 => add_6
# add_6 => add_8
# any_2 => any_2
# clamp => clamp_max, clamp_min, convert_element_type_8, convert_element_type_9
# isinf_1 => isinf_1
# l__self___block_0_layer_0_dropout => clone_3
# l__self___block_0_layer__1__dropout => clone_5
# l__self___embed_tokens => embedding
# neg => neg
# where_1 => full_default_1, full_default_2, where_1
triton_red_fused_add_any_clamp_clone_embedding_isinf_neg_scalar_tensor_where_8 = async_compile.triton('triton_red_fused_add_any_clamp_clone_embedding_isinf_neg_scalar_tensor_where_8', '''
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
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*i1', 4: '*fp16', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_any_clamp_clone_embedding_isinf_neg_scalar_tensor_where_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_any_clamp_clone_embedding_isinf_neg_scalar_tensor_where_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = (rindex // 512)
        r1 = rindex % 512
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (16*x0)), xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r3 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp15 = tl.load(in_ptr3 + (r3 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 32128, tmp0)
        # tl.device_assert(((0 <= tmp1) & (tmp1 < 32128)) | ~xmask, "index out of bounds: 0 <= tmp1 < 32128")
        tmp2 = tl.load(in_ptr1 + (r1 + (512*tmp1)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = 64504.0
        tmp9 = 65504.0
        tmp10 = tl.where(tmp7, tmp8, tmp9)
        tmp11 = -tmp10
        tmp12 = triton_helpers.maximum(tmp5, tmp11)
        tmp13 = triton_helpers.minimum(tmp12, tmp10)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.isinf(tmp16).to(tl.int1)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 | tmp18
        _tmp19 = tl.where(xmask, tmp20, _tmp19)
        tl.store(in_out_ptr0 + (r3 + (8192*x0)), tmp16, xmask)
    tmp19 = triton_helpers.any(_tmp19.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''')
