

# Original file: ./BlenderbotForCausalLM__73_backward_290.34/BlenderbotForCausalLM__73_backward_290.34_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/ch/cchnapvoo462bbj5qydhhx6ikzpz2sseqvuqk7sedsdcxtckiei3.py
# Source Nodes: [add_1, l__self___final_layer_norm], Original ATen: [aten._to_copy, aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_1 => add_3
# l__self___final_layer_norm => mul_5, sub_2
triton_red_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_6 = async_compile.triton('triton_red_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_6', '''
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
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr3 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp7 + tmp9
        tmp12 = tmp10 - tmp11
        tmp14 = tmp12 * tmp13
        tmp15 = tmp3 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask & xmask, tmp18, _tmp17)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr6 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp22 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp29 = tl.load(in_ptr3 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp39 = tl.load(in_ptr7 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp20 = 2560.0
        tmp21 = tmp13 / tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp25 = tmp23 * tmp24
        tmp26 = tmp25 * tmp20
        tmp27 = tmp26 - tmp5
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp28 + tmp30
        tmp32 = tmp31 - tmp11
        tmp33 = tmp32 * tmp13
        tmp34 = tmp33 * tmp17
        tmp35 = tmp27 - tmp34
        tmp36 = tmp21 * tmp35
        tmp37 = tmp19 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp40 = tmp39.to(tl.float32)
        tmp41 = 1.1111111111111112
        tmp42 = tmp40 * tmp41
        tmp43 = tmp38 * tmp42
        tl.store(out_ptr2 + (r1 + (2560*x0)), tmp37, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (2560*x0)), tmp43, rmask & xmask)
''')
