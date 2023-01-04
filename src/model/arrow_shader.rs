pub const BILLBOARD_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

struct VectorInput {
    color_1: f32,
    color_2: f32,
    color_3: f32,
    orig_1: f32,
    orig_2: f32,
    orig_3: f32,
    arrow_1: f32,
    arrow_2: f32,
    arrow_3: f32,
}

struct BillboardTransform {
    transform: mat3x3<f32>,
//    _padding: vec3<f32>,
}

@group(1) @binding(0)
var<storage, read> vector_inputs: array<VectorInput>;
@group(1) @binding(1)
var<storage, read_write> transforms: array<BillboardTransform>;

@compute @workgroup_size(256, 1, 1)
fn cp_main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
    let total = arrayLength(&transforms);
    let index = global_invocation_id.x + global_invocation_id.y * u32(64);
    if (index >= total) {
        return;
    }
    let vector_i = vector_inputs[index];
    let orig_position = vec3<f32>(vector_i.orig_1, vector_i.orig_2, vector_i.orig_3);
    let arrow = vec3<f32>(vector_i.arrow_1, vector_i.arrow_2, vector_i.arrow_3);
    let view_axis = normalize(orig_position - camera.view_pos.xyz);
    let arrow_axis = normalize(arrow);
    let right_axis = normalize(cross(view_axis, arrow_axis));
    let rotation_mat = mat3x3<f32>(
        right_axis,
        arrow_axis,
        view_axis);
    //let rotation_mat = mat3x3<f32>(1., 0., 0., 0., 1., 0., 0., 0., 1.);
    transforms[index].transform = rotation_mat;
}
";
pub const ARROW_SHADER: &str = "
// Vertex shader

// Define any uniforms we expect from app
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

/*
struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}
*/

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

/*
@group(1) @binding(0)
var<uniform> transform: TransformUniform;
*/

// This is the input from the vertex buffer we created
// We get the properties from our Vertex struct here
// Note the index on location -- this relates to the properties placement in the buffer stride
// e.g. 0 = 1st \"set\" of data, 1 = 2nd \"set\"
struct VertexInput {
    @location(0) position: vec3<f32>,
    //@location(1) color: vec3<f32>,
    //@location(2) coords: vec2<f32>,
    //@location(3) orig_position: vec3<f32>,
    //@location(4) arrow: vec3<f32>,
};

struct VectorInput {
    @location(1) color: vec3<f32>,
    //@location(2) coords: vec2<f32>,
    @location(2) orig_position: vec3<f32>,
    @location(3) arrow: vec3<f32>,

};

// The output we send to our fragment shader
struct VertexOutput {
    // This property is \"builtin\" (aka used to render our vertex shader)
    @builtin(position) clip_position: vec4<f32>,
    // These are \"custom\" properties we can create to pass down
    // In this case, we pass the color down
    @location(0) color: vec3<f32>,
    //@location(1) coords: vec2<f32>,
};

struct BillboardTransform {
    @location(4) t_1: vec3<f32>,
    @location(5) t_2: vec3<f32>,
    @location(6) t_3: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    vector_i: VectorInput,
    t: BillboardTransform,
) -> VertexOutput {
    //let model_matrix = transform.model;
    //let normal_matrix = transform.normal;

    // We define the output we want to send over to frag shader
    var out: VertexOutput;

    //out.coords = vec2<f32>(model.coords.y - 0.5, model.coords.x - 0.5);
    out.color = vector_i.color;
    let view_axis = normalize(vector_i.orig_position - camera.view_pos.xyz);
    let arrow_axis = normalize(vector_i.arrow);
    let right_axis = normalize(cross(view_axis, arrow_axis));
    let rotation_mat = mat3x3<f32>(
        right_axis,
        arrow_axis,
        view_axis);
        /*
    let rotation_mat = mat3x3<f32>(
        t.t_1,
        t.t_2,
        t.t_3);
        */
    let position = rotation_mat * (model.position - vec3<f32>(0.5, 0., 0.)) * 0.1 + vec3<f32>(vector_i.orig_position);

    // We set the \"position\" by using the `clip_position` property
    // We multiply it by the camera position matrix and the instance position matrix
    //out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    return out;
}

// Fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // We use the special function `textureSample` to combine the texture data with coords
    let arrow_color = vec4<f32>(in.color, 1.);
    /*
    let bg_color = vec4<f32>(0., 0., 0., 0.);
    let mix_arrow_value_t = arrow_triangle(in.coords, 1., .2, .2, 0.005, 0.001);
    let mix_arrow_value = 1. - smoothstep(0.005, 0.01, mix_arrow_value_t);
    let result = mix(bg_color, arrow_color, mix_arrow_value);
    */
    let result = arrow_color;

    return result;
}
";
