

# Original file: ./Super_SloMo___60.0/Super_SloMo___60.0_1.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/float32/l6/cl6hoor7227negojgtccaxcqlys7xphbtnpekwcnlvlkjclhzb3y.py
# Source Nodes: [g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___l1_loss_fn], Original ATen: [aten.abs, aten.mean, aten.sub]
# g____import_torchbenchmark_dot_models_dot_super_slo_mo_dot_model_wrapper___l1_loss_fn => abs_1, mean, sub_109
triton_per_fused_abs_mean_sub_34 = async_compile.triton('triton_per_fused_abs_mean_sub_34', '''
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
    size_hints=[1, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_abs_mean_sub_34', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]}
)
@triton.jit
def triton_per_fused_abs_mean_sub_34(in_ptr0, out_ptr0, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 273
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = triton_helpers.promote_to_tensor(tl.sum(tmp3, 0))
    tl.store(out_ptr0 + (tl.full([1], 0, tl.int32)), tmp4, None)
''')
