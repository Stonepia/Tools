

# Original file: ./hf_T5_large___60.0/hf_T5_large___60.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/tg/ctgdfikdsnix3peq7mm5ppeckjyla6nk3xwhwimaua3o4cxid5wd.py
# Source Nodes: [add_157, add_159, any_76, clamp_73, clamp_74, isinf_75, l__mod___model_decoder_block_8_layer__1__dropout, l__mod___model_decoder_block_9_layer_0_dropout, neg_74, neg_75, where_1, where_75, where_76], Original ATen: [aten.add, aten.any, aten.clamp, aten.clone, aten.isinf, aten.neg, aten.scalar_tensor, aten.where]
# add_157 => add_200
# add_159 => add_203
# any_76 => any_76
# clamp_73 => clamp_max_73, clamp_min_73, convert_element_type_389, convert_element_type_390
# clamp_74 => clamp_max_74, clamp_min_74, convert_element_type_393, convert_element_type_394
# isinf_75 => isinf_75
# l__mod___model_decoder_block_8_layer__1__dropout => clone_194
# l__mod___model_decoder_block_9_layer_0_dropout => clone_197
# neg_74 => neg_74
# neg_75 => neg_75
# where_1 => full_default_2, full_default_3
# where_75 => where_75
# where_76 => where_76
triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_17 = async_compile.triton('triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_17', '''
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
    size_hints=[64, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: '*i1', 5: '*i1', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_17', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp14 = tl.load(in_ptr3 + (0))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_out_ptr0 + (r1 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = 64504.0
        tmp5 = 65504.0
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = -tmp6
        tmp8 = triton_helpers.maximum(tmp1, tmp7)
        tmp9 = triton_helpers.minimum(tmp8, tmp6)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12.to(tl.float32)
        tmp16 = tl.where(tmp15, tmp4, tmp5)
        tmp17 = -tmp16
        tmp18 = triton_helpers.maximum(tmp13, tmp17)
        tmp19 = triton_helpers.minimum(tmp18, tmp16)
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.isinf(tmp22).to(tl.int1)
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 | tmp24
        _tmp25 = tl.where(xmask, tmp26, _tmp25)
        tl.store(in_out_ptr0 + (r1 + (8192*x0)), tmp22, xmask)
    tmp25 = triton_helpers.any(_tmp25.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp25, xmask)
''')