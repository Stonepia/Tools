

# Original file: ./pit_b_224___60.0/pit_b_224___60.0_2.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/timm_models/amp_fp16/pf/cpf3ejt7qebucw4fh4lsqrekab7avunha4ocgw3oq25k3mmwabya.py
# Source Nodes: [add_15, add_16, add_17, add_18, getattr_l__self___transformers_1_blocks___4___attn_proj_drop, getattr_l__self___transformers_1_blocks___4___mlp_drop2, getattr_l__self___transformers_1_blocks___5___attn_proj_drop, getattr_l__self___transformers_1_blocks___5___mlp_drop2], Original ATen: [aten.add, aten.clone]
# add_15 => add_52
# add_16 => add_56
# add_17 => add_59
# add_18 => add_63
# getattr_l__self___transformers_1_blocks___4___attn_proj_drop => clone_22
# getattr_l__self___transformers_1_blocks___4___mlp_drop2 => clone_24
# getattr_l__self___transformers_1_blocks___5___attn_proj_drop => clone_25
# getattr_l__self___transformers_1_blocks___5___mlp_drop2 => clone_27
triton_poi_fused_add_clone_25 = async_compile.triton('triton_poi_fused_add_clone_25', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clone_25', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]})
@triton.jit
def triton_poi_fused_add_clone_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8421376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(in_out_ptr0 + (x0), tmp8, None)
''')