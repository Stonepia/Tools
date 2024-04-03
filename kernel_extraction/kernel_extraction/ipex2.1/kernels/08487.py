

# Original file: ./DistillGPT2__0_backward_99.1/DistillGPT2__0_backward_99.1_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/huggingface/amp_fp16/bh/cbhiq3ljxtz3bb4g6wqcxir237kyxfvvju5ci5yvorls2uk6d5f5.py
# Source Nodes: [add_17, l__self___transformer_h_4_ln_1, l__self___transformer_h_4_ln_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# add_17 => add_35
# l__self___transformer_h_4_ln_1 => mul_58, sub_12
# l__self___transformer_h_4_ln_2 => mul_64, sub_14
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_25 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_25', '''
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=())]}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 768
    x1 = (xindex // 768)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp25 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp8 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr5 + (x0 + (768*r2) + (98304*x1)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp18 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 - tmp6
        tmp9 = tmp7 * tmp8
        tmp10 = tmp1 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp4 - tmp16
        tmp19 = tmp17 * tmp18
        tmp20 = tmp15 * tmp19
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp24 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp26 = _tmp25 + tmp24
        _tmp25 = tl.where(rmask, tmp26, _tmp25)
        tmp27 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp22, None)
    tmp25 = tl.sum(_tmp25, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp25, None)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp28, None)
''')
