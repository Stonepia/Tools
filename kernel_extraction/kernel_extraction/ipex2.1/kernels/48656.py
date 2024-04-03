

# Original file: ./BlenderbotForCausalLM__40_backward_301.45/BlenderbotForCausalLM__40_backward_301.45_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/qk/cqkkre2y7cy3d74hdm2runzu6dlxnf2amouffocvpwo35spildvd.py
# Source Nodes: [l__self___final_layer_norm], Original ATen: [aten.add, aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___final_layer_norm => convert_element_type_4
triton_red_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_red_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*i1', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_dropout_backward_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp16 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp9 = tmp8.to(tl.float32)
        tmp11 = tmp9 - tmp10
        tmp13 = tmp11 * tmp12
        tmp14 = tmp4 * tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
        tmp17 = _tmp16 + tmp15
        _tmp16 = tl.where(rmask & xmask, tmp17, _tmp16)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp16 = tl.sum(_tmp16, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp18 = tl.load(in_ptr5 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr6 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first')
        tmp19 = 2560.0
        tmp20 = tmp12 / tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp23.to(tl.float32)
        tmp25 = tmp22 * tmp24
        tmp26 = tmp25 * tmp19
        tmp27 = tmp26 - tmp6
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp29 - tmp10
        tmp31 = tmp30 * tmp12
        tmp32 = tmp31 * tmp16
        tmp33 = tmp27 - tmp32
        tmp34 = tmp20 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp18 + tmp35
        tmp38 = tmp37.to(tl.float32)
        tmp39 = 1.1111111111111112
        tmp40 = tmp38 * tmp39
        tmp41 = tmp36 * tmp40
        tl.store(out_ptr2 + (r1 + (2560*x0)), tmp36, rmask & xmask)
        tl.store(out_ptr3 + (r1 + (2560*x0)), tmp41, rmask & xmask)
''')
