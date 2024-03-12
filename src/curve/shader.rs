use super::CurveData;
use crate::shader;

macro_rules! SHADER { () => {"
struct CameraUniform {{
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}}

struct Light {{
    position: vec3<f32>,
    color: vec3<f32>,
}}

struct TransformUniform {{
    model: mat4x4<f32>,
    normal: mat4x4<f32>,
}}

struct Jitter {{
    jitter: vec4<f32>,
}}

struct SettingsUniform {{
    radius: f32,
    color: vec3<f32>,
}}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;
@group(0) @binding(2)
var<uniform> jitter: Jitter;

@group(1) @binding(0)
var<uniform> transform: TransformUniform;

@group(2) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {{
    @location(0) position: vec3<f32>,
}};

struct PosInput {{
    @location(1) position_1: vec3<f32>,
    @location(2) position_2: vec3<f32>,
}};

// Data Input
{}

{}

struct VertexOutput {{
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos_1: vec3<f32>,
	@location(1) world_pos_2: vec3<f32>,
	@location(2) world_pos: vec3<f32>,
    // Data Ouput
    {}
}};

@vertex
fn vs_main(
    model: VertexInput,
    pos: PosInput,
    {}
) -> VertexOutput {{
    let model_matrix = transform.model;
    //let center_vector = (model_matrix * vec4<f32>(pos.position_2 - pos.position_1, 1.)).xyz;
    let center_vector = pos.position_2 - pos.position_1;

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let view_axis = normalize((model_matrix * vec4<f32>(pos.position_1, 1.)).xyz - camera.view_pos.xyz);
    let camera_up = normalize(cross(center_vector, view_axis));
    //let camera_up = normalize(vec3<f32>(camera.view_proj.x.y, camera.view_proj.y.y, camera.view_proj.z.y));
    let world_position = (model_matrix * vec4<f32>(pos.position_1 + (0.5*(model.position.x + 1.) * center_vector + model.position.y * camera_up * settings.radius), 1.)).xyz;
    let clip_pos = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.clip_position = clip_pos + jitter.jitter * clip_pos.w;
    out.world_pos_1 = (model_matrix * vec4<f32>(pos.position_1, 1.)).xyz;
    out.world_pos_2 = (model_matrix * vec4<f32>(pos.position_2, 1.)).xyz;
    out.world_pos = world_position;
    let t = 0.5 * (model.position.x + 1.);

    // Output set
    {}
    return out;
}}

// cylinder defined by extremes a and b, and radious ra
fn cylIntersect( ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32 ) -> vec4<f32>
{{
    let ba = b  - a;
    let oc = ro - a;
    let baba = dot(ba,ba);
    let bard = dot(ba,rd);
    let baoc = dot(ba,oc);
    let k2 = baba            - bard*bard;
    let k1 = baba*dot(oc,rd) - baoc*bard;
    let k0 = baba*dot(oc,oc) - baoc*baoc - ra*ra*baba;
    var h = k1*k1 - k2*k0;
    if( h<0.0 ) {{ return vec4(-1.0); }}//no intersection
    h = sqrt(h);
    let t = (-k1-h)/k2;
    // body
    let y = baoc + t*bard;
    if( y>0.0 && y<baba ) {{ return vec4( t, (oc+t*rd - ba*y/baba)/ra ); }}
    return vec4(-1.0);//no intersection
}}

// normal at sphere p of cylinder (a,b,ra), see above
fn cylNormal( p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32 ) -> vec3<f32>
{{
    let pa = p - a;
    let ba = b - a;
    let baba = dot(ba,ba);
    let paba = dot(pa,ba);
    let h = dot(pa,ba)/baba;
    return (pa - ba*h)/ra;
}}

struct FragOutput {{
    @builtin(frag_depth) depth: f32,
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
}}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {{
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
	let a = in.world_pos_1;
	let b = in.world_pos_2;
    let det = determinant(transform.normal);
    let r = settings.radius / pow(det, 1. / 3.);
	let t = cylIntersect(ro, rd, a, b, r);

    var out: FragOutput;

	let pos = ro + t.x * rd;
	let normal = cylNormal(pos, a, b, r);

    {}

    let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.albedo = vec4<f32>(lambertian, 0.3);
    out.normal = vec4<f32>((normal + vec3<f32>(1.)) / 2. , 0.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}}
"};}

pub fn get_shader(data_format: Option<&CurveData>) -> String {
    let data_struct = match data_format {
        Some(CurveData::Scalar(..)) => {
            "
struct DataInput {
    @location(3) val_1: f32,
    @location(4) val_2: f32,
}"
        }
        Some(CurveData::Color(_)) => {
            "
struct DataInput {
    @location(3) val_1: vec3<f32>,
    @location(4) val_2: vec3<f32>,
}"
        }
        None => "",
    };

    let uniform = match data_format {
        Some(CurveData::Scalar(..)) => shader::COLORMAP_UNIFORM,
        _ => "",
    };

    let output_val = match data_format {
        Some(CurveData::Scalar(..)) => "@location(3) val: f32,",
        Some(CurveData::Color(_)) => "@location(3) val: vec3<f32>",
        None => "",
    };

    let input_val = match data_format {
        Some(_) => "data: DataInput,",
        None => "",
    };

    let set_output = match data_format {
        Some(_) => "out.val = data.val_1 * (1. - t) + t * data.val_2;",
        None => "",
    };

    let color_output = match data_format {
        Some(CurveData::Scalar(..)) => "let lambertian = colormap(in.val);",
        Some(CurveData::Color(_)) => "let lambertian = in.val;",
        None => "let lambertian = settings.color;",
    };

    format!(
        SHADER!(),
        data_struct, uniform, output_val, input_val, set_output, color_output,
    )
}
