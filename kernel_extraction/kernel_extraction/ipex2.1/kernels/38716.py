

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/qt/cqthycmbasre4ab7mtmdqhn342b3iny3sod6bcpntxofp4znbck7.py
# Source Nodes: [g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___mse_loss_fn], Original ATen: [aten.mse_loss]
# g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___mse_loss_fn => mean_1, pow_1
triton_per_fused_mse_loss_43 = async_compile.triton('triton_per_fused_mse_loss_43', '''
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
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mse_loss_43', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_mse_loss_43(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 46464
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 192
    x1 = (xindex // 192)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((512*((r2 + (128*x0) + (24576*x1)) % 1936)) + (991232*((r2 + (128*x0) + (24576*x1)) // 991232)) + (((r2 + (128*x0) + (24576*x1)) // 1936) % 512)), rmask & xmask, other=0.0)
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''')
