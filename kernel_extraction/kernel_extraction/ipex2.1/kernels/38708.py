

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/lk/clkbwus6tpr6m4i3twkz3dllvwo7qrkgdpnj75zv5s7dqdydt2r4.py
# Source Nodes: [g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___l1_loss_fn_1, g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___l1_loss_fn_2], Original ATen: [aten.abs, aten.mean, aten.sub]
# g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___l1_loss_fn_1 => abs_2, mean_2, sub_111
# g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___l1_loss_fn_2 => abs_3, mean_3, sub_112
triton_per_fused_abs_mean_sub_35 = async_compile.triton('triton_per_fused_abs_mean_sub_35', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_mean_sub_35', 'configs': [AttrsDescriptor(divisible_by_16=(1, 3, 4, 5, 6), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_abs_mean_sub_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 17472
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 64
    x1 = (xindex // 64)
    x3 = xindex
    tmp0 = r2 + (128*x0)
    tmp1 = tl.full([1, 1], 8170, tl.int32)
    tmp2 = tmp0 < tmp1
    tmp3 = r2 + (128*x0) + (8170*x1)
    tmp4 = tl.full([1, 1], 2230272, tl.int32)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp5 & tmp2
    tmp7 = tl.load(in_ptr0 + ((20*((r2 + (128*x0) + (8170*x1)) % 123904)) + (2478080*(((r2 + (128*x0) + (8170*x1)) // 371712) % 6)) + (((r2 + (128*x0) + (8170*x1)) // 123904) % 3)), rmask & tmp6 & xmask, other=0.0)
    tmp8 = tl.load(in_ptr1 + ((r2 + (128*x0) + (8170*x1)) % 2230272), rmask & tmp6 & xmask, other=0.0)
    tmp9 = tmp7 - tmp8
    tmp10 = tl.abs(tmp9)
    tmp11 = tl.where(tmp6, tmp10, 0)
    tmp12 = tl.where(tmp2, tmp11, 0)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.load(in_ptr2 + ((20*((r2 + (128*x0) + (8170*x1)) % 123904)) + (2478080*(((r2 + (128*x0) + (8170*x1)) // 371712) % 6)) + (((r2 + (128*x0) + (8170*x1)) // 123904) % 3)), rmask & tmp6 & xmask, other=0.0)
    tmp18 = tmp17 - tmp8
    tmp19 = tl.abs(tmp18)
    tmp20 = tl.where(tmp6, tmp19, 0)
    tmp21 = tl.where(tmp2, tmp20, 0)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp24 = tl.where(rmask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp16, xmask)
    tl.store(out_ptr1 + (x3), tmp25, xmask)
''')
