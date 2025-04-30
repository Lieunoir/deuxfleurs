use crate::data::*;
use crate::texture;
use crate::types::{Color, Scalar};
use crate::ui::UiDataElement;
use crate::updater::*;
use crate::util;
use crate::util::Vertex;
use wgpu::util::DeviceExt;

mod picker;
mod shader;
use picker::Picker;
use shader::get_shader;

pub enum PointCloudData {
    Scalar(Vec<f32>, ColorMap),
    Color(Vec<[f32; 3]>),
}

impl DataSettings for PointCloudData {
    fn apply_settings(&mut self, other: Self) {
        match (self, other) {
            (PointCloudData::Scalar(_, set1), PointCloudData::Scalar(_, set2)) => *set1 = set2,
            _ => (),
        }
    }
}

impl UiDataElement for PointCloudData {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool {
        match self {
            PointCloudData::Scalar(_, settings) => settings.draw(ui, property_changed),
            PointCloudData::Color(_) => false,
        }
    }
}

impl DataUniformBuilder for PointCloudData {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        match self {
            PointCloudData::Scalar(_, colormap) => colormap.get_value().build_uniform(device),
            _ => None,
        }
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        match self {
            PointCloudData::Scalar(_, colormap) => {
                colormap.get_value().refresh_buffer(queue, data_uniform)
            }
            _ => (),
        }
    }
}

impl PointCloudData {
    fn sphere_desc<'a>(&self) -> wgpu::VertexBufferLayout<'a> {
        match self {
            PointCloudData::Color(_) => SphereColorData::desc(),
            PointCloudData::Scalar(..) => SphereScalarData::desc(),
        }
    }

    fn build_sphere_data_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        match self {
            PointCloudData::Scalar(scalars, _) => {
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
            PointCloudData::Color(colors) => {
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

#[repr(C)]
#[derive(Copy, Clone, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PCSettings {
    radius: Radius,
    color: ColorSettings,
}

impl UiDataElement for PCSettings {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool {
        let changed = self.radius.draw(ui, property_changed);
        self.color.draw(ui, property_changed) || changed
    }
}

impl NamedSettings for PCSettings {
    fn set_name(mut self, name: &str) -> Self {
        self.color = ColorSettings::new(name);
        self
    }
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

pub struct PointCloudGeometry {
    pub positions: Vec<[f32; 3]>,
    pub num_elements: u32,
}

impl Positions for PointCloudGeometry {
    fn get_positions(&self) -> &[[f32; 3]] {
        &self.positions
    }
}

pub(crate) struct PointCloudFixedRenderer {
    positions_len: u32,
    vertex_buffer: wgpu::Buffer,
    center_buffer: wgpu::Buffer,
}

pub(crate) struct PointCloudDataBuffer {
    sphere_data_buffer: Option<wgpu::Buffer>,
}

pub(crate) struct PointCloudPipeline {
    sphere_render_pipeline: wgpu::RenderPipeline,
}

impl DataBuffer for PointCloudDataBuffer {
    type Settings = PCSettings;
    type Data = PointCloudData;
    type Geometry = PointCloudGeometry;

    fn new(device: &wgpu::Device, _geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self {
        let sphere_data_buffer = data.map(|d| d.build_sphere_data_buffer(device));
        Self { sphere_data_buffer }
    }
}

impl FixedRenderer for PointCloudFixedRenderer {
    type Settings = PCSettings;
    type Data = PointCloudData;
    type Geometry = PointCloudGeometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self {
        let s2 = 2_f32.sqrt();
        //let s2 = 1.;
        let positions = [[-s2, -s2, 0.], [s2, -s2, 0.], [-s2, s2, 0.], [s2, s2, 0.]];
        let vertices = positions.map(|position| SphereVertex { position });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut gpu_vertices = Vec::with_capacity(geometry.positions.len());
        for position in geometry.positions.iter() {
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
            positions_len: geometry.positions.len() as u32,
        }
    }
}

impl RenderPipeline for PointCloudPipeline {
    type Settings = PCSettings;
    type Data = PointCloudData;
    type Geometry = PointCloudGeometry;
    type Fixed = PointCloudFixedRenderer;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        _fixed: &Self::Fixed,
        _settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let bind_group_layouts = if let Some(uniform) = data_uniform {
            vec![
                camera_light_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout,
                &uniform.bind_group_layout,
            ]
        } else {
            vec![
                camera_light_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout,
            ]
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sphere cloud Render Pipeline Layout"),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("sphere cloud shader"),
            source: wgpu::ShaderSource::Wgsl(get_shader(data).into()),
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
        let sphere_render_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &sphere_buffer_layout,
            shader,
            Some("cloud sphere render"),
        );
        PointCloudPipeline {
            sphere_render_pipeline,
        }
    }
}

type PointCloudRenderer =
    Renderer<PointCloudFixedRenderer, PointCloudDataBuffer, PointCloudPipeline>;

impl Render for PointCloudRenderer {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.transform_uniform.bind_group, &[]);
        render_pass.set_bind_group(2, &self.settings_uniform.bind_group, &[]);
        if let Some(uniform) = &self.data_uniform {
            render_pass.set_bind_group(3, &uniform.bind_group, &[]);
        }
        render_pass.set_pipeline(&self.pipeline.sphere_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.center_buffer.slice(..));
        if let Some(data_buffer) = &self.data_buffer.sphere_data_buffer {
            render_pass.set_vertex_buffer(2, data_buffer.slice(..));
        }
        //render_pass.draw(0..6, 0..(self.fixed.positions_len));
        render_pass.draw(0..4, 0..(self.fixed.positions_len));
    }
}

pub type PointCloud = BareElement<PCSettings, PointCloudData, PointCloudGeometry>;

pub(crate) type DisplayPointCloud = DisplayElement<
    PCSettings,
    PointCloudData,
    PointCloudGeometry,
    PointCloudFixedRenderer,
    PointCloudDataBuffer,
    PointCloudPipeline,
    Picker,
>;

impl DisplayPointCloud {
    pub(crate) fn new(
        name: String,
        positions: Vec<[f32; 3]>,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let element = PointCloud::new(name, positions);
        DisplayPointCloud::init(
            element,
            device,
            camera_light_bind_group_layout,
            counter_bind_group_layout,
            color_format,
        )
    }
}

impl PointCloud {
    pub(crate) fn new(name: String, positions: Vec<[f32; 3]>) -> Self {
        let geometry = PointCloudGeometry {
            num_elements: positions.len() as u32,
            positions,
        };
        PointCloud::init(name, geometry)
    }

    pub fn set_radius(&mut self, radius: f32) -> &mut Self {
        self.updater.settings.radius.radius = radius;
        self.updater.settings_changed = true;
        self
    }

    pub fn set_color(&mut self, color: [f32; 4]) -> &mut Self {
        self.updater.settings.color.color = color;
        self.updater.settings_changed = true;
        self
    }

    pub fn add_scalar<S: Scalar>(&mut self, name: String, datas: S) -> &mut PointCloudData {
        let datas = datas.into();
        assert!(datas.len() == self.geometry().positions.len());
        let settings = ColorMap::new(&datas);
        self.updater
            .add_data(name, PointCloudData::Scalar(datas, settings))
    }

    pub fn add_colors<C: Color>(&mut self, name: String, datas: C) -> &mut PointCloudData {
        let datas = datas.into();
        assert!(datas.len() == self.geometry().positions.len());
        self.updater.add_data(name, PointCloudData::Color(datas))
    }

    pub(crate) fn draw_element_info(&self, element: usize, ui: &mut egui::Ui) {
        if element < self.geometry().positions.len() {
            ui.label(format!("Picked point number {}", element));
        }
    }
}
