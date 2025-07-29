struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    min_bb: vec2<f32>,
    max_bb: vec2<f32>,
    shadow_proj: mat4x4<f32>,
}


struct Jitter {
    jitter: vec4<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}


@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> jitter: Jitter;

@group(1) @binding(0)
var<uniform> transform: TransformUniform;


// This is the input from the vertex buffer we created
// We get the properties from our Vertex struct here
// Note the index on location -- this relates to the properties placement in the buffer stride
// e.g. 0 = 1st \"set\" of data, 1 = 2nd \"set\"
struct VertexInput {
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) face_normal: vec4<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
) -> @builtin(position) vec4<f32> {
    let clip_pos = camera.shadow_proj * transform.model * vec4<f32>(model.position, 1.0);
    return clip_pos + jitter.jitter * clip_pos.w;
    //return clip_pos;
}

// Fragment shader
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) f32 {
    return 1.;
}
