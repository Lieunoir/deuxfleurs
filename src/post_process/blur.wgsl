@group(0) @binding(0)
var t_a: texture_2d<f32>;
@group(0) @binding(1)
var s: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

const pos = array(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0));

@vertex
fn vs_main(
    @builtin(vertex_index) index: u32,
) -> VertexOutput {

    var out: VertexOutput;
    let clip_pos = vec4<f32>(pos[index].x * 2. - 1., pos[index].y * 2. - 1., 0., 1.);
    out.clip_position = clip_pos;
    out.tex_coords = pos[index];
    return out;
}

const OFFSET = array(0.0, 1.3846153846, 3.2307692308);
const WEIGHT = array(0.2270270270, 0.3162162162, 0.0702702703);

@fragment
fn horizontal_fs_main(in: VertexOutput) -> @location(0) f32 {
    let buffer_size = vec2<f32>(textureDimensions(t_a));
    let coords = in.tex_coords;
    var weight = textureSample(t_a, s, coords).x * WEIGHT[0];
    let dpx = buffer_size.x;
    for (var r = 1; r < 3; r++) {
        weight += textureSample(t_a, s, coords + vec2<f32>(OFFSET[r] / dpx, 0.)).x * WEIGHT[r];
        weight += textureSample(t_a, s, coords - vec2<f32>(OFFSET[r] / dpx, 0.)).x * WEIGHT[r];
    }
    return weight;
}

@fragment
fn vertical_fs_main(in: VertexOutput) -> @location(0) f32 {
    let buffer_size = vec2<f32>(textureDimensions(t_a));
    let coords = in.tex_coords;
    var weight = textureSample(t_a, s, coords).x * WEIGHT[0];
    let dpy = buffer_size.y;
    for (var r = 1; r < 3; r++) {
        weight += textureSample(t_a, s, coords + vec2<f32>(0., OFFSET[r] / dpy)).x * WEIGHT[r];
        weight += textureSample(t_a, s, coords - vec2<f32>(0., OFFSET[r] / dpy)).x * WEIGHT[r];
    }
    return weight;
}
