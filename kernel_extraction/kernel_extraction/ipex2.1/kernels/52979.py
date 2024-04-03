

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_bf16/ig/cigl3lgufywy7cbrewtv63i2q7ekfzhfxnnzqbcopkm2s75bx3iv.py
# Source Nodes: [l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__self___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_11 => convert_element_type_221
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8', '''
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
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr3 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp10 - tmp11
        tmp14 = tmp12 * tmp13
        tmp15 = tmp5 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tmp18
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp21 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
        tmp29 = tl.load(in_ptr3 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp19 = 4096.0
        tmp20 = tmp13 / tmp19
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tmp27 = tmp26 * tmp19
        tmp28 = tmp27 - tmp7
        tmp30 = tmp29.to(tl.float32)
        tmp31 = tmp30 - tmp11
        tmp32 = tmp31 * tmp13
        tmp33 = tmp32 * tmp17
        tmp34 = tmp28 - tmp33
        tmp35 = tmp20 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp36, None)
''')
