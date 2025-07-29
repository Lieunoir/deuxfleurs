@group(0) @binding(0)
var t_copy: texture_2d<f32>;

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
    //return log(textureLoad(t_copy, vec2<i32>(floor(in.clip_position.xy)), 0).x);
    return 0.0008740438 / (- 1.0001 + textureLoad(t_copy, vec2<i32>(floor(in.clip_position.xy)), 0).x);
    //return - 0.0008740438 / (1144.2218 - textureLoad(t_copy, vec2<i32>(floor(in.clip_position.xy)), 0).x);
}


//{"eye":[0.045391057,0.26075774,-2.4887884],"target":[0.045391057,0.26075774,0.13308074],"up":[0.0,1.0,0.0]}
