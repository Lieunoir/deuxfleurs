struct Parameters {
    reprojection: mat4x4<f32>,
    depth_mul: f32,
    depth_add: f32,
    x_mul: f32,
    x_add: f32,
    y_mul: f32,
    y_add: f32,
    frame_index: u32,
    _pad: u32,
}

@group(0) @binding(0)
var t_copy: texture_2d<f32>;
@group(0) @binding(1)
var<uniform> param: Parameters;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

const pos = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(pos[index], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    return param.depth_mul / (param.depth_add + textureLoad(t_copy, vec2<i32>(in.clip_position.xy), 0).x);
}
