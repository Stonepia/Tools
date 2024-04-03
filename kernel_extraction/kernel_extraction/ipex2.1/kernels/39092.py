

# Original file: ./DebertaV2ForMaskedLM__0_backward_207.1/DebertaV2ForMaskedLM__0_backward_207.1_3.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/km/ckm5itwwesytsihfxej6qvmi264ij3iv4k3pab6pojkc56y6qqc3.py
# Source Nodes: [gelu_24, l__self___cls_predictions_transform_layer_norm], Original ATen: [aten.gelu, aten.gelu_backward, aten.native_layer_norm, aten.native_layer_norm_backward]
# gelu_24 => add_171, convert_element_type_701, erf_24, mul_270
# l__self___cls_predictions_transform_layer_norm => convert_element_type_703
triton_red_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_red_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5', '''
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
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_gelu_gelu_backward_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp7 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask & xmask, tmp6, _tmp5)
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 - tmp9
        tmp12 = tmp10 * tmp11
        tmp13 = tmp3 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask & xmask, tmp16, _tmp15)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp19 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
        tmp25 = tl.load(in_ptr2 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr5 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = 1536.0
        tmp18 = tmp11 / tmp17
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = tmp22 * tmp17
        tmp24 = tmp23 - tmp5
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp26 - tmp9
        tmp28 = tmp27 * tmp11
        tmp29 = tmp28 * tmp15
        tmp30 = tmp24 - tmp29
        tmp31 = tmp18 * tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = 0.7071067811865476
        tmp35 = tmp33 * tmp34
        tmp36 = libdevice.erf(tmp35)
        tmp37 = 1.0
        tmp38 = tmp36 + tmp37
        tmp39 = 0.5
        tmp40 = tmp38 * tmp39
        tmp41 = tmp33 * tmp33
        tmp42 = -0.5
        tmp43 = tmp41 * tmp42
        tmp44 = tl.exp(tmp43)
        tmp45 = 0.3989422804014327
        tmp46 = tmp44 * tmp45
        tmp47 = tmp33 * tmp46
        tmp48 = tmp40 + tmp47
        tmp49 = tmp31 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp50, rmask & xmask)
''')
