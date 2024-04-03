

# Original file: ./LayoutLMForSequenceClassification__0_backward_171.1/LayoutLMForSequenceClassification__0_backward_171.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/fj/cfjwkchizhcwjil2qialachlz44qsaagadujtldxjbyl5e44crmn.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.native_dropout_backward, aten.native_layer_norm_backward, aten.select_backward, aten.slice_backward]

triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_select_backward_slice_backward_9 = async_compile.triton('triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_select_backward_slice_backward_9', '''
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
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i1', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_select_backward_slice_backward_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused__to_copy_native_dropout_backward_native_layer_norm_backward_select_backward_slice_backward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp5 = 0.0
        tmp6 = tl.where(tmp2, tmp4, tmp5)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp13 = tmp8 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tl.load(in_ptr3 + (x3), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp21 = tl.load(in_ptr0 + (r2 + (768*x1)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr2 + (r2 + (768*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp35 = tl.load(in_ptr4 + (r2 + (768*x3)), rmask, eviction_policy='evict_first')
        tmp18 = x0
        tmp19 = tl.full([1, 1], 0, tl.int32)
        tmp20 = tmp18 == tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = 0.0
        tmp24 = tl.where(tmp20, tmp22, tmp23)
        tmp26 = tmp24 * tmp25
        tmp27 = 768.0
        tmp28 = tmp26 * tmp27
        tmp29 = tmp28 - tmp10
        tmp31 = tmp30 * tmp15
        tmp32 = tmp29 - tmp31
        tmp33 = tmp17 * tmp32
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = 1.1111111111111112
        tmp38 = tmp36 * tmp37
        tmp39 = tmp34 * tmp38
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp33, rmask)
        tl.store(out_ptr3 + (r2 + (768*x3)), tmp39, rmask)
''')
