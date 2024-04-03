

# Original file: ./BlenderbotForCausalLM__79_backward_288.32/BlenderbotForCausalLM__79_backward_288.32_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ap/capqb7h7ey345yeaeh47oryuk7gya7jcaojmqwwog7ifjjo2mda5.py
# Source Nodes: [l__self___self_attn_layer_norm], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___self_attn_layer_norm => convert_element_type
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    _tmp20 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp5 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
        tmp13 = tmp12.to(tl.float32)
        tmp15 = tmp13 - tmp14
        tmp17 = tmp15 * tmp16
        tmp18 = tmp8 * tmp17
        tmp19 = tl.broadcast_to(tmp18, [XBLOCK, RBLOCK])
        tmp21 = _tmp20 + tmp19
        _tmp20 = tl.where(rmask & xmask, tmp21, _tmp20)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp20 = tl.sum(_tmp20, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp34 = tl.load(in_ptr4 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp40 = tl.load(in_out_ptr0 + (r1 + (2560*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 * tmp29
        tmp31 = 2560.0
        tmp32 = tmp30 * tmp31
        tmp33 = tmp32 - tmp10
        tmp35 = tmp34.to(tl.float32)
        tmp36 = tmp35 - tmp14
        tmp37 = tmp36 * tmp16
        tmp38 = tmp37 * tmp20
        tmp39 = tmp33 - tmp38
        tmp41 = tmp16 / tmp31
        tmp42 = tmp41 * tmp39
        tmp43 = tmp42.to(tl.float32)
        tmp44 = tmp40 + tmp43
        tl.store(in_out_ptr0 + (r1 + (2560*x0)), tmp44, rmask & xmask)
''')
