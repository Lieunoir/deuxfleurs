use crate::data::*;
use crate::texture;
use crate::types::{Color, Scalar};
use crate::updater::*;
use crate::util;
use wgpu::util::DeviceExt;

pub enum CloudData {
    Scalar(Vec<f32>, ColorMap),
    Color(Vec<[f32; 3]>),
}

impl DataUniformBuilder for CloudData {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        match self {
            CloudData::Scalar(_, colormap) => colormap.get_value().build_uniform(device),
            _ => None,
        }
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        match self {
            CloudData::Scalar(_, colormap) => colormap.get_value().refresh_buffer(queue, data_uniform),
            _ => (),
        }
    }
}

impl CloudData {
    fn sphere_desc<'a>(&self) -> wgpu::VertexBufferLayout<'a> {
        match self {
            CloudData::Color(_) => SphereColorData::desc(),
            CloudData::Scalar(..) => SphereScalarData::desc(),
        }
    }

    fn build_sphere_data_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        match self {
            CloudData::Scalar(scalars, _) => {
                let mut min_d = scalars[0];
                let mut max_d = scalars[0];
                for data in scalars {
                    if *data > max_d {
                        max_d = *data;
                    }
                    if *data < min_d {
                        min_d = *data;
                    }
                }
                let gpu_vertices: Vec<_> = scalars
                    .iter()
                    .map(|data| {
                        let t = (data - min_d) / (max_d - min_d);
                        SphereScalarData { scalar: t }
                    })
                    .collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Cloud Sphere Center Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            CloudData::Color(colors) => {
                let gpu_vertices: Vec<_> = colors
                    .iter()
                    .map(|color| SphereColorData { color: *color })
                    .collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Cloud Sphere Center Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }
}

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PCSettings {
    pub radius: Radius,
    pub color: ColorSettings,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereVertex {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereCenter {
    position: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereColorData {
    color: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereScalarData {
    scalar: f32,
}

impl Vertex for SphereVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl Vertex for SphereCenter {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl Vertex for SphereColorData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x3,
            }],
        }
    }
}

impl Vertex for SphereScalarData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32,
            }],
        }
    }
}

pub struct CloudItem {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub num_elements: u32,
}

pub struct CloudFixedRenderer {
    positions_len: u32,
    vertex_buffer: wgpu::Buffer,
    center_buffer: wgpu::Buffer,
}

pub struct CloudDataBuffer {
    sphere_data_buffer: Option<wgpu::Buffer>,
}

pub struct CloudPipeline {
    sphere_render_pipeline: wgpu::RenderPipeline,
}

impl DataBuffer for CloudDataBuffer {
    type Settings = PCSettings;
    type Data = CloudData;
    type Item = CloudItem;

    fn new(device: &wgpu::Device, item: &Self::Item, data: Option<&Self::Data>) -> Self {
        let sphere_data_buffer = data.map(|d| d.build_sphere_data_buffer(device));
        Self {
            sphere_data_buffer,
        }
    }
}

impl FixedRenderer for CloudFixedRenderer {
    type Settings = PCSettings;
    type Data = CloudData;
    type Item = CloudItem;

    fn initialize(device: &wgpu::Device, item: &Self::Item, settings: &Self::Settings) -> Self {
        //let s2 = 2_f32.sqrt();
        let s2 = 1.;
        let positions = [
            [-s2, -s2, 0.],
            [s2, -s2, 0.],
            [-s2, s2, 0.],
            [s2, -s2, 0.],
            [s2, s2, 0.],
            [-s2, s2, 0.],
        ];
        let vertices = positions.map(|position| SphereVertex { position });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut gpu_vertices = Vec::with_capacity(item.positions.len());
        for position in item.positions.iter() {
            let vertex = SphereCenter {
                position: *position,
            };
            gpu_vertices.push(vertex);
        }
        let center_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cloud Sphere Center Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            vertex_buffer,
            center_buffer,
            positions_len: item.positions.len() as u32,
        }
    }
}

impl RenderPipeline for CloudPipeline {
    type Settings = PCSettings;
    type Data = CloudData;
    type Item = CloudItem;
    type Fixed = CloudFixedRenderer;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        fixed: &Self::Fixed,
        settings: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let bind_group_layouts = if let Some(uniform) = data_uniform{
            vec![camera_light_bind_group_layout, &settings.bind_group_layout, &uniform.bind_group_layout]
        } else {
            vec![camera_light_bind_group_layout, &settings.bind_group_layout]
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sphere cloud Render Pipeline Layout"),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("sphere cloud shader"),
            source: wgpu::ShaderSource::Wgsl(get_shader(data.is_some()).into()),
        };
        let sphere_buffer_layout = if let Some(data) = &data {
            vec![
                SphereVertex::desc(),
                SphereCenter::desc(),
                data.sphere_desc(),
            ]
        } else {
            vec![SphereVertex::desc(), SphereCenter::desc()]
        };
        let sphere_render_pipeline = util::create_render_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &sphere_buffer_layout,
            shader,
            Some("cloud sphere render"),
        );
        CloudPipeline {
            sphere_render_pipeline,
        }
    }
}

pub type CloudRenderer =
    Renderer<PCSettings, CloudData, CloudItem, CloudFixedRenderer, CloudDataBuffer, CloudPipeline>;

impl Render for CloudRenderer {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.settings_uniform.bind_group, &[]);
        if let Some(uniform) = &self.data_uniform {
            render_pass.set_bind_group(2, &uniform.bind_group, &[]);
        }
        render_pass.set_pipeline(&self.pipeline.sphere_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.center_buffer.slice(..));
        if let Some(data_buffer) = &self.data_buffer.sphere_data_buffer {
            render_pass.set_vertex_buffer(2, data_buffer.slice(..));
        }
        render_pass.draw(0..6, 0..(self.fixed.positions_len));
    }
}

pub type PointCloud = MainDisplayItem<
    PCSettings,
    CloudData,
    CloudItem,
    CloudFixedRenderer,
    CloudDataBuffer,
    CloudPipeline,
>;

impl PointCloud {
    pub fn new(
        name: String,
        positions: Vec<[f32; 3]>,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        //transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let item = CloudItem {
            name,
            num_elements: positions.len() as u32,
            positions,
        };
        PointCloud::init(device, item, camera_light_bind_group_layout, color_format)
    }

    pub fn set_radius(&mut self, radius: f32) {
        self.updater.settings.radius.radius = radius;
        self.updater.settings_changed = true;
    }

    pub fn set_color(&mut self, color: [f32; 4]) {
        self.updater.settings.color.color = color;
        self.updater.settings_changed = true;
    }

    pub fn add_scalar<S: Scalar>(&mut self, name: String, datas: S) -> &mut Self {
        let datas = datas.into();
        assert!(datas.len() == self.item.positions.len());
        if let Some(data_name) = &mut self.updater.shown_data {
            if *data_name == name {
                self.updater.data_to_show = Some(Some(data_name.clone()));
                self.updater.shown_data = None;
            }
        }
        //TODO recover old settings
        let settings = ColorMap::default();
        self.updater.data.insert(name, CloudData::Scalar(datas, settings));
        self
    }

    pub fn add_colors<C: Color>(&mut self, name: String, datas: C) -> &mut Self {
        let datas = datas.into();
        assert!(datas.len() == self.item.positions.len());
        if let Some(data_name) = &mut self.updater.shown_data {
            if *data_name == name {
                self.updater.data_to_show = Some(Some(data_name.clone()));
                self.updater.shown_data = None;
            }
        }
        self.updater.data.insert(name, CloudData::Color(datas));
        self
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.updater.data_to_show = Some(name);
        self
    }
}

fn get_shader(show_data: bool) -> &'static str {
    if !show_data {
        SPHERE_SHADER
    } else {
        SPHERE_DATA_SHADER
    }
}

//macro_rules! SHADER { () => {"
const SPHERE_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

//struct TransformUniform {
//    model: mat4x4<f32>,
//    normal: mat4x4<f32>,
//}

struct SettingsUniform {
    radius: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

//@group(1) @binding(0)
//var<uniform> transform: TransformUniform;
@group(1) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct DataInput {
    @location(1) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
	//@location(2) arrow: vec3<f32>,
	//@location(3) radius: f32,
};

@vertex
fn vs_main(
    model: VertexInput,
    data: DataInput,
) -> VertexOutput {
    //let model_matrix = transform.model;
    //let normal_matrix = transform.normal;

    //let world_vector_pos = (model_matrix * vec4<f32>(vector_i.orig_position, 1.)).xyz;
    //// Do we want to scale a vector field if we scale its attached mesh?
    //let world_vector_arrow_t = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;
    //let arrow_ampl = length(world_vector_arrow_t);
    //let world_vector_arrow = normalize(world_vector_arrow_t);

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj.x.x, camera.view_proj.y.x, camera.view_proj.z.x));
    let camera_up = normalize(vec3<f32>(camera.view_proj.x.y, camera.view_proj.y.y, camera.view_proj.z.y));
    let world_position = data.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos = world_position;
    out.center = data.position;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }


const PI: f32 = 3.14159;

// PBR functions taken from https://learnopengl.com/PBR/Theory
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {
    let a2     = a*a;
    let NdotH  = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;
	
    let nom    = a2;
    var denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;
	
    return nom / denom;
}

fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32
{
    let nom   = NdotV;
    let denom = NdotV * (1.0 - k) + k;
	
    return nom / denom;
}
  
fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32
{
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = GeometrySchlickGGX(NdotV, k);
    let ggx2 = GeometrySchlickGGX(NdotL, k);
	
    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32>
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn sphIntersect( ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32 ) -> vec2<f32>
{
    let oc = ro - ce;
    let b = dot( oc, rd );
    let c = dot( oc, oc ) - ra*ra;
    var h = b*b - c;
    if( h<0.0 ) { return vec2<f32>(-1.0); } // no intersection
    h = sqrt( h );
    return vec2<f32>( -b-h, -b+h );
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let ce = in.center;
    let r = settings.radius;
    //let pa = in.orig_position;
    //let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
    //let pb2 = in.orig_position + in.arrow * 0.1;

    var out: FragOutput;

    let t = sphIntersect( ro, rd, ce, r);
    if(t.x < 0.0) {
        discard;
    }
	let pos = ro + t.x * rd;
	let normal = normalize(pos - ce);

	let light_dir = normalize(light.position - pos);
	let view_dir = normalize(camera.view_pos.xyz - pos);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.15);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let lambertian = settings.color;
	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	out.color = vec4<f32>(result, 1.);
	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}
";

const SPHERE_DATA_SHADER: &str = "
struct CameraUniform {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}

//struct TransformUniform {
//    model: mat4x4<f32>,
//    normal: mat4x4<f32>,
//}

struct SettingsUniform {
    radius: f32,
    color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniform;
@group(0) @binding(1)
var<uniform> light: Light;

//@group(1) @binding(0)
//var<uniform> transform: TransformUniform;
@group(1) @binding(0)
var<uniform> settings: SettingsUniform;
struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct DataInput {
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
	@location(0) world_pos: vec3<f32>,
	@location(1) center: vec3<f32>,
	@location(2) color: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    data: DataInput,
) -> VertexOutput {
    //let model_matrix = transform.model;
    //let normal_matrix = transform.normal;

    //let world_vector_pos = (model_matrix * vec4<f32>(vector_i.orig_position, 1.)).xyz;
    //// Do we want to scale a vector field if we scale its attached mesh?
    //let world_vector_arrow_t = (model_matrix * vec4<f32>(vector_i.orig_position + vector_i.arrow, 1.)).xyz - world_vector_pos;
    //let arrow_ampl = length(world_vector_arrow_t);
    //let world_vector_arrow = normalize(world_vector_arrow_t);

    //// We define the output we want to send over to frag shader
    var out: VertexOutput;

    let camera_right = normalize(vec3<f32>(camera.view_proj.x.x, camera.view_proj.y.x, camera.view_proj.z.x));
    let camera_up = normalize(vec3<f32>(camera.view_proj.x.y, camera.view_proj.y.y, camera.view_proj.z.y));
    let world_position = data.position + (model.position.x * camera_right + model.position.y * camera_up) * settings.radius;
    out.clip_position = camera.view_proj * vec4<f32>(world_position, 1.0);
    out.world_pos = world_position;
    out.center = data.position;
    out.color = data.color;
    return out;
}

// function from :
// https://iquilezles.org/articles/intersectors/
fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }


const PI: f32 = 3.14159;

// PBR functions taken from https://learnopengl.com/PBR/Theory
fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {
    let a2     = a*a;
    let NdotH  = max(dot(N, H), 0.0);
    let NdotH2 = NdotH*NdotH;
	
    let nom    = a2;
    var denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    denom        = PI * denom * denom;
	
    return nom / denom;
}

fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32
{
    let nom   = NdotV;
    let denom = NdotV * (1.0 - k) + k;
	
    return nom / denom;
}
  
fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32
{
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let ggx1 = GeometrySchlickGGX(NdotV, k);
    let ggx2 = GeometrySchlickGGX(NdotL, k);
	
    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32>
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

fn sphIntersect( ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32 ) -> vec2<f32>
{
    let oc = ro - ce;
    let b = dot( oc, rd );
    let c = dot( oc, oc ) - ra*ra;
    var h = b*b - c;
    if( h<0.0 ) { return vec2<f32>(-1.0); } // no intersection
    h = sqrt( h );
    return vec2<f32>( -b-h, -b+h );
}

struct FragOutput {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
}

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let ro = camera.view_pos.xyz;
	let rd = normalize(in.world_pos - camera.view_pos.xyz);
    let ce = in.center;
    let r = settings.radius;
    //let pa = in.orig_position;
    //let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
    //let pb2 = in.orig_position + in.arrow * 0.1;

    var out: FragOutput;

    let t = sphIntersect( ro, rd, ce, r);
    if(t.x < 0.0) {
        discard;
    }
	let pos = ro + t.x * rd;
	let normal = normalize(pos - ce);

	let light_dir = normalize(light.position - pos);
	let view_dir = normalize(camera.view_pos.xyz - pos);
	let half_dir = normalize(view_dir + light_dir);
	let F0 = vec3<f32>(0.04, 0.04, 0.04);
	let D = DistributionGGX(normal, half_dir, 0.15);
	let F = fresnelSchlick(dot(half_dir, normal), F0);
	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
	let kd = 1.0;
	let lambertian = in.color;
	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
	out.color = vec4<f32>(result, 1.);
	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
	out.depth = clip_space_pos.z / clip_space_pos.w;
	return out;
}
";
//// function from :
//// https://iquilezles.org/articles/intersectors/
//fn dot2(v: vec3<f32>) -> f32 { return dot(v, v); }
//
//
//const PI: f32 = 3.14159;
//
//// PBR functions taken from https://learnopengl.com/PBR/Theory
//fn DistributionGGX(N: vec3<f32>, H: vec3<f32>, a: f32) -> f32 {
//    let a2     = a*a;
//    let NdotH  = max(dot(N, H), 0.0);
//    let NdotH2 = NdotH*NdotH;
//	
//    let nom    = a2;
//    var denom  = (NdotH2 * (a2 - 1.0) + 1.0);
//    denom        = PI * denom * denom;
//	
//    return nom / denom;
//}
//
//fn GeometrySchlickGGX(NdotV: f32, k: f32) -> f32
//{
//    let nom   = NdotV;
//    let denom = NdotV * (1.0 - k) + k;
//	
//    return nom / denom;
//}
//  
//fn GeometrySmith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, k: f32) -> f32
//{
//    let NdotV = max(dot(N, V), 0.0);
//    let NdotL = max(dot(N, L), 0.0);
//    let ggx1 = GeometrySchlickGGX(NdotV, k);
//    let ggx2 = GeometrySchlickGGX(NdotL, k);
//	
//    return ggx1 * ggx2;
//}
//
//fn fresnelSchlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32>
//{
//    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
//}
//
//fn sphIntersect( ro: vec3<f32>, rd: vec3<f32>, ce: vec3<f32>, ra: f32 ) -> vec2<f32>
//{
//    let oc = ro - ce;
//    let b = dot( oc, rd );
//    let c = dot( oc, oc ) - ra*ra;
//    var h = b*b - c;
//    if( h<0.0 ) { return vec2<f32>(-1.0); } // no intersection
//    h = sqrt( h );
//    return vec2<f32>( -b-h, -b+h );
//}
//
//struct FragOutput {
//    @builtin(frag_depth) depth: f32,
//    @location(0) color: vec4<f32>,
//}
//
//@fragment
//fn fs_main(in: VertexOutput) -> FragOutput {
//    let ro = camera.view_pos.xyz;
//	let rd = normalize(in.world_pos - camera.view_pos.xyz);
//    let ce = in.center;
//    let r = settings.radius;
//    //let pa = in.orig_position;
//    //let pb1 = in.orig_position + 0.5 * in.arrow * 0.1;
//    //let pb2 = in.orig_position + in.arrow * 0.1;
//
//    var out: FragOutput;
//
//    let t = sphIntersect( ro, rd, ce, r);
//    if(t.x < 0.0) {
//        discard;
//    }
//	let pos = ro + t.x * rd;
//	let normal = normalize(pos - ce);
//
//	let light_dir = normalize(light.position - pos);
//	let view_dir = normalize(camera.view_pos.xyz - pos);
//	let half_dir = normalize(view_dir + light_dir);
//	let F0 = vec3<f32>(0.04, 0.04, 0.04);
//	let D = DistributionGGX(normal, half_dir, 0.15);
//	let F = fresnelSchlick(dot(half_dir, normal), F0);
//	let G = GeometrySmith(normal, view_dir, light_dir, 0.01);
//	let f_ct = D * F * G / (4. * dot(view_dir, normal) * dot(light_dir, normal));
//	let kd = 1.0;
//	let lambertian = in.color;
//	let result = (kd * lambertian + PI * f_ct) * light.color * max(dot(normal, light_dir), 0.0);
//	out.color = vec4<f32>(result, 1.);
//	let clip_space_pos = camera.view_proj * vec4<f32>(pos, 1.);
//	out.depth = clip_space_pos.z / clip_space_pos.w;
//	return out;
//}
//";
