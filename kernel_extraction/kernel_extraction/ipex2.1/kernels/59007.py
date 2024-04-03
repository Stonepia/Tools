

# Original file: ./AllenaiLongformerBase__22_backward_143.5/AllenaiLongformerBase__22_backward_143.5.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float32/pb/cpb4gowg7gfowix5membfbuzobqqul4l7tpqbvk3l3ldp5xdma4z.py
# Source Nodes: [], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm_backward]

triton_red_fused_native_dropout_backward_native_layer_norm_backward_31 = async_compile.triton('triton_red_fused_native_dropout_backward_native_layer_norm_backward_31', '''
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
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_dropout_backward_native_layer_norm_backward_31', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_dropout_backward_native_layer_norm_backward_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp2 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (768*x1) + (3072*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(rmask, tmp3, _tmp2)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp10 = tl.load(in_ptr0 + (r2 + (768*x1) + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0)
        tmp14 = tl.load(in_ptr1 + (r2 + (768*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp18 = tl.load(in_ptr3 + (r2 + (768*x3)), rmask, eviction_policy='evict_first')
        tmp11 = 768.0
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12 - tmp2
        tmp15 = tmp14 * tmp7
        tmp16 = tmp13 - tmp15
        tmp17 = tmp9 * tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = 1.1111111111111112
        tmp21 = tmp19 * tmp20
        tmp22 = tmp17 * tmp21
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp17, rmask)
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp22, rmask)
''')
