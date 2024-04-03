

# Original file: ./hf_T5_base___60.0/hf_T5_base___60.0_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float16/q7/cq7eor34py3kkzacbmmrrwdgalomshywb5tyhq5kt3mgfcmvefnj.py
# Source Nodes: [add_8, any_3, clamp_1, isinf_2, l__mod___model_encoder_block_1_layer_0_dropout, neg_1, where_1, where_2], Original ATen: [aten.add, aten.any, aten.clamp, aten.clone, aten.isinf, aten.neg, aten.scalar_tensor, aten.where]
# add_8 => add_11
# any_3 => any_3
# clamp_1 => clamp_max_1, clamp_min_1, convert_element_type_12, convert_element_type_13
# isinf_2 => isinf_2
# l__mod___model_encoder_block_1_layer_0_dropout => clone_8
# neg_1 => neg_1
# where_1 => full_default_2, full_default_3
# where_2 => where_2
triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_13 = async_compile.triton('triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_13', '''
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
    size_hints=[256, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i1', 2: '*fp16', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_any_clamp_clone_isinf_neg_scalar_tensor_where_13(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 192
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8192*x0)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr2 + (r1 + (8192*x0)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp4 = 64504.0
        tmp5 = 65504.0
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = -tmp6
        tmp8 = triton_helpers.maximum(tmp1, tmp7)
        tmp9 = triton_helpers.minimum(tmp8, tmp6)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = libdevice.isinf(tmp12).to(tl.int1)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 | tmp14
        _tmp15 = tl.where(xmask, tmp16, _tmp15)
    tmp15 = triton_helpers.any(_tmp15.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr0 + (x0), tmp15, xmask)
''')
