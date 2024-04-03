

# Original file: ./LayoutLMForSequenceClassification__0_backward_135.1/LayoutLMForSequenceClassification__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/bfloat16/ze/czethufrfe72wc4iprkm7zuxr4hmqv3ci7gjj52xfsyxl6ijas7r.py
# Source Nodes: [l__mod___layoutlm_encoder_layer_11_output_layer_norm], Original ATen: [aten.native_dropout_backward, aten.native_layer_norm, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]
# l__mod___layoutlm_encoder_layer_11_output_layer_norm => convert_element_type_97
triton_red_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5 = async_compile.triton('triton_red_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5', '''
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
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*i1', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_native_dropout_backward_native_layer_norm_native_layer_norm_backward_select_backward_slice_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    tmp15 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x3), None, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = 0.0
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 - tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp9 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp28 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp45 = tl.load(in_ptr5 + (r2 + (768*x3)), rmask, eviction_policy='evict_first')
        tmp23 = 768.0
        tmp24 = tmp17 / tmp23
        tmp25 = x0
        tmp26 = tl.full([1, 1], 0, tl.int32)
        tmp27 = tmp25 == tmp26
        tmp29 = 0.0
        tmp30 = tl.where(tmp27, tmp28, tmp29)
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp31 * tmp33
        tmp35 = tmp34 * tmp23
        tmp36 = tmp35 - tmp11
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp38 - tmp15
        tmp40 = tmp39 * tmp17
        tmp41 = tmp40 * tmp21
        tmp42 = tmp36 - tmp41
        tmp43 = tmp24 * tmp42
        tmp44 = tmp43.to(tl.float32)
        tmp46 = tmp45.to(tl.float32)
        tmp47 = 1.1111111111111112
        tmp48 = tmp46 * tmp47
        tmp49 = tmp44 * tmp48
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp44, rmask)
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp49, rmask)
''')
