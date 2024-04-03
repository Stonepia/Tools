

# Original file: ./T5Small__0_forward_169.0/T5Small__0_forward_169.0_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/ug/cugrj4j4ktsbxhmh77d2vqwaxxaea27gluzfyrbhbcpsxqmauobs.py
# Source Nodes: [add_4, any_1, isinf, l__mod___encoder_block_0_layer_0_dropout, l__mod___encoder_dropout], Original ATen: [aten.add, aten.any, aten.isinf, aten.native_dropout]
# add_4 => add_6
# any_1 => any_1
# isinf => isinf
# l__mod___encoder_block_0_layer_0_dropout => gt_3, mul_10, mul_9
# l__mod___encoder_dropout => mul_1, mul_2
triton_red_fused_add_any_isinf_native_dropout_6 = async_compile.triton('triton_red_fused_add_any_isinf_native_dropout_6', '''
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
    meta={'signature': {0: '*i64', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*i1', 5: '*i1', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_any_isinf_native_dropout_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_any_isinf_native_dropout_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp19 = tl.full([XBLOCK, RBLOCK], 0, tl.int1)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr1 + (r1 + (8192*x0)), xmask, eviction_policy='evict_first')
        tmp8 = tl.load(in_ptr2 + (r1 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr3 + (r1 + (8192*x0)), xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp0 = tl.load(in_ptr0 + load_seed_offset)
        tmp1 = r1 + (8192*x0)
        tmp2 = tl.rand(tmp0, (tmp1).to(tl.uint32))
        tmp3 = tmp2.to(tl.float32)
        tmp4 = 0.1
        tmp5 = tmp3 > tmp4
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = 1.1111111111111112
        tmp11 = tmp9 * tmp10
        tmp12 = tmp5.to(tl.float32)
        tmp14 = tmp12 * tmp13
        tmp15 = tmp14 * tmp10
        tmp16 = tmp11 + tmp15
        tmp17 = libdevice.isinf(tmp16).to(tl.int1)
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp20 = _tmp19 | tmp18
        _tmp19 = tl.where(xmask, tmp20, _tmp19)
        tl.store(out_ptr1 + (r1 + (8192*x0)), tmp5, xmask)
    tmp19 = triton_helpers.any(_tmp19.to(tl.int8), 1)[:, None].to(tl.int1)
    tl.store(out_ptr2 + (x0), tmp19, xmask)
''')
