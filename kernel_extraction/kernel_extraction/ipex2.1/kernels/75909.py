

# Original file: ./detectron2_maskrcnn__28_inference_68.8/detectron2_maskrcnn__28_inference_68.8_0.py

import torch
import intel_extension_for_pytorch
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
async_compile = XPUAsyncCompile()

# kernel path: /home/sdp/tongsu/pytorch/inductor_log/torchbench/bfloat16/xi/cxi242btyzwl5bugpfvmq5tck2cjvtfuaobcfgxwwf5t7epz7pdm.py
# Source Nodes: [interpolate_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.clone]
# interpolate_1 => _unsafe_index_1, clone_1, convert_element_type_11, convert_element_type_6
triton_poi_fused__to_copy__unsafe_index_clone_1 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_clone_1', '''
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_clone_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3891200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 38912)
    x1 = (xindex // 256) % 152
    x0 = xindex % 256
    x4 = xindex
    tmp0 = x2
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.5
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x1
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = tmp12 * tmp6
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.load(in_ptr0 + (x0 + (256*tmp14) + (19456*tmp8)), None).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp17, None)
''')
