

# Original file: ./AlbertForQuestionAnswering__0_backward_135.1/AlbertForQuestionAnswering__0_backward_135.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/float16/vj/cvjjth42tdj3b5fdw6ocngwiucvyxvdyg62yr5xwvpbgzuxkdilm.py
# Source Nodes: [l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# l__mod___albert_encoder_albert_layer_groups_0_albert_layers_0_attention_layer_norm_11 => convert_element_type_71
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp10 = tl.load(in_ptr3 + (r1 + (4096*x0)), None, eviction_policy='evict_last').to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp13 = tmp11 - tmp12
        tmp15 = tmp13 * tmp14
        tmp16 = tmp6 * tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tmp19
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr0 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last').to(tl.float32)
        tmp31 = tl.load(in_ptr3 + (r1 + (4096*x0)), None, eviction_policy='evict_first').to(tl.float32)
        tmp20 = 4096.0
        tmp21 = tmp14 / tmp20
        tmp24 = tmp22 + tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp29 = tmp28 * tmp20
        tmp30 = tmp29 - tmp8
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32 - tmp12
        tmp34 = tmp33 * tmp14
        tmp35 = tmp34 * tmp18
        tmp36 = tmp30 - tmp35
        tmp37 = tmp21 * tmp36
        tmp38 = tmp37.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (4096*x0)), tmp38, None)
''')
