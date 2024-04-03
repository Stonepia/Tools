

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/c5/cc5ppmczjgea6jc4revrzovt3kl6pfqsdpudmulfwvqxqmvlfato.py
# Source Nodes: [l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_full_layer_layer_norm_10 => convert_element_type_210
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
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r1), None, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tmp12
        tl.store(out_ptr0 + (r1 + (4096*x0)), tmp9, None)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp16 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last')
        tmp14 = tl.load(in_ptr5 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp15 - tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp13 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tmp23
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp26 = tl.load(out_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first')
        tmp29 = tl.load(in_ptr5 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp24 = 4096.0
        tmp25 = tmp18 / tmp24
        tmp27 = tmp26 * tmp24
        tmp28 = tmp27 - tmp11
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp16
        tmp32 = tmp31 * tmp18
        tmp33 = tmp32 * tmp22
        tmp34 = tmp28 - tmp33
        tmp35 = tmp25 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr3 + (r1 + (4096*x0)), tmp36, None)
''')
