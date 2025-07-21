struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    min_bb: vec2<f32>,
    max_bb: vec2<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

struct GroundLevel {
    level: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(1) @binding(0)
var t_a: texture_2d<f32>;
@group(1) @binding(1)
var s: sampler;
@group(1) @binding(2)
var<uniform> level: GroundLevel;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

const pos = array(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index : u32,
    ) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = vec4<f32>(camera.min_bb.x + pos[index].x * camera.max_bb.x, level.level, camera.min_bb.y + pos[index].y * camera.max_bb.y, 1.);

    out.clip_position = camera.view_proj * world_pos;
    out.tex_coords = vec2<f32>(pos[index].x, 1. - pos[index].y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let weight = textureSample(t_a, s, in.tex_coords).x;
    //These values are in linear color space and should NOT correspond to the screen output values
    return vec4<f32>(0., 0., 0., 0.4 * weight);
}
