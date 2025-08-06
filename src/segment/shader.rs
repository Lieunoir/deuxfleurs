use super::geometry::SegmentData;
use crate::shader;

macro_rules! SHADER { () => {"
struct CameraUniform {{
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}}

struct TransformUniform {{
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
    scale: f32,
}}

struct Jitter {{
    jitter: vec4<f32>,
}}

struct SettingsUniform {{
    radius: f32,
    char_len: f32,
    color: vec3<f32>,
}}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
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
	@location(0) view_pos_1: vec3<f32>,
	@location(1) view_pos_2: vec3<f32>,
	@location(2) view_pos: vec3<f32>,
    // Data Ouput
    {}
}};

@vertex
fn vs_main(
    model: VertexInput,
    pos: PosInput,
    {}
) -> VertexOutput {{
    let model_matrix = camera.view * transform.model;
    let view_pos_1 = (model_matrix * vec4<f32>(pos.position_1, 1.)).xyz;
    let view_pos_2 = (model_matrix * vec4<f32>(pos.position_2, 1.)).xyz;
    let center_vector = view_pos_2 - view_pos_1;
    //let center_vector = pos.position_2 - pos.position_1;

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let view_axis = normalize(view_pos_1);
    let camera_up = normalize(cross(center_vector, view_axis));
    let view_position = view_pos_1 + 0.5*(model.position.x + 1.) * center_vector + model.position.y * camera_up * settings.radius * settings.char_len * transform.scale;
    let clip_pos = camera.proj * vec4<f32>(view_position, 1.0);
    out.clip_position = clip_pos + jitter.jitter * clip_pos.w;
    out.view_pos_1 = view_pos_1;
    out.view_pos_2 = view_pos_2;
    out.view_pos = view_position;
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
    let ro = vec3<f32>(0.);
	let rd = normalize(in.view_pos);
	let a = in.view_pos_1;
	let b = in.view_pos_2;
    let det = determinant(transform.normal);
    let r = settings.radius * settings.char_len / pow(det, 1. / 3.);
	let t = cylIntersect(ro, rd, a, b, r);

    var out: FragOutput;

	let pos = ro + t.x * rd;
	let normal = cylNormal(pos, a, b, r);

    {}

    let clip_space_pos = camera.proj * vec4<f32>(pos, 1.);
	out.albedo = vec4<f32>(lambertian, 0.3);
    out.normal = vec4<f32>((normal + vec3<f32>(1.)) / 2. , 0.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}}
"};}

pub fn get_shader(data_format: Option<&SegmentData>) -> String {
    let data_struct = match data_format {
        Some(SegmentData::Scalar(..)) => {
            "
struct DataInput {
    @location(3) val_1: f32,
    @location(4) val_2: f32,
}"
        }
        Some(SegmentData::Color(_)) => {
            "
struct DataInput {
    @location(3) val_1: vec3<f32>,
    @location(4) val_2: vec3<f32>,
}"
        }
        None => "",
    };

    let uniform = match data_format {
        Some(SegmentData::Scalar(..)) => shader::COLORMAP_UNIFORM,
        _ => "",
    };

    let output_val = match data_format {
        Some(SegmentData::Scalar(..)) => "@location(3) val: f32,",
        Some(SegmentData::Color(_)) => "@location(3) val: vec3<f32>",
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
        Some(SegmentData::Scalar(..)) => "let lambertian = colormap(in.val);",
        Some(SegmentData::Color(_)) => "let lambertian = in.val;",
        None => "let lambertian = settings.color;",
    };

    format!(
        SHADER!(),
        data_struct, uniform, output_val, input_val, set_output, color_output,
    )
}

pub const CYLINDER_PICKER_SHADER: &str = "
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
}

struct TransformUniform {
    model: mat4x4<f32>,
    normal: mat3x3<f32>,
    scale: f32,
}

struct CounterUniform {
    count: u32,
    _padding_1: u32,
    _padding_2: u32,
    _padding_3: u32,
}

struct SettingsUniform {
    radius: f32,
    char_len: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var<uniform> counter: CounterUniform;

@group(2) @binding(0)
var<uniform> transform: TransformUniform;
@group(3) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct PosInput {
    @location(1) position_1: vec3<f32>,
    @location(2) position_2: vec3<f32>,
    @builtin(instance_index) index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) view_pos_1: vec3<f32>,
	@location(1) view_pos_2: vec3<f32>,
	@location(2) view_pos: vec3<f32>,
    @location(3) index: u32,
};

override offset: u32 = 0;

@vertex
fn vs_main(
    model: VertexInput,
    pos: PosInput,
) -> VertexOutput {
    let model_matrix = camera.view * transform.model;
    let view_pos_1 = (model_matrix * vec4<f32>(pos.position_1, 1.)).xyz;
    let view_pos_2 = (model_matrix * vec4<f32>(pos.position_2, 1.)).xyz;
    let center_vector = view_pos_2 - view_pos_1;
    //let center_vector = pos.position_2 - pos.position_1;

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let view_axis = normalize(view_pos_1);
    let camera_up = normalize(cross(center_vector, view_axis));
    let view_position = view_pos_1 + 0.5*(model.position.x + 1.) * center_vector + model.position.y * camera_up * settings.radius * settings.char_len * transform.scale;
    out.clip_position = camera.proj * vec4<f32>(view_position, 1.0);
    out.view_pos_1 = view_pos_1;
    out.view_pos_2 = view_pos_2;
    out.view_pos = view_position;
    out.index = counter.count + pos.index + offset;
    let t = 0.5 * (model.position.x + 1.);
    return out;
}

// cylinder defined by extremes a and b, and radious ra
fn cylIntersect( ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32 ) -> vec4<f32>
{
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
}

// normal at sphere p of cylinder (a,b,ra), see above
fn cylNormal( p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, ra: f32 ) -> vec3<f32>
{
    let pa = p - a;
    let ba = b - a;
    let baba = dot(ba,ba);
    let paba = dot(pa,ba);
    let h = dot(pa,ba)/baba;
    return (pa - ba*h)/ra;
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = vec3<f32>(0.);
	let rd = normalize(in.view_pos);
	let a = in.view_pos_1;
	let b = in.view_pos_2;
    let r = settings.radius * settings.char_len * transform.scale;
	let t = cylIntersect(ro, rd, a, b, r);

    var out: FragOutput;

	let pos = ro + t.x * rd;

    let clip_space_pos = camera.proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
    let res = in.index;
    // webgl dosen't support rendering to u32, so we have to resort to this
    let f1 = f32((res >> u32(24))) / 255.;
    let f2 = f32(((res << u32(8)) >> u32(24))) / 255.;
    let f3 = f32(((res << u32(16)) >> u32(24))) / 255.;
    let f4 = f32(((res << u32(24)) >> u32(24))) / 255.;
    out.color = vec4<f32>(f4, f3, f2, f1);
	return out;
}";
