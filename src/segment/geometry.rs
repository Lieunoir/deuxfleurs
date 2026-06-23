use crate::data::{internal::*, *};
use crate::picker::SegmentPicked;
use crate::shape::*;
use crate::texture;
use crate::types::{Color, Scalar};
use crate::util;
use crate::util::Vertex;
use crate::window::{ContextHolder, InnerBareState, InnerGraphicalState};
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub enum SegmentData {
    Scalar(Vec<f32>, ColorMap),
    Color(Vec<[f32; 3]>),
}

impl DataSettings for SegmentData {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        match self {
            SegmentData::Scalar(_, settings) => settings.draw_ui(ui),
            SegmentData::Color(_) => false,
        }
    }

    fn apply_previous_settings(&mut self, other: Self) {
        match (self, other) {
            (SegmentData::Scalar(_, set1), SegmentData::Scalar(_, set2)) => {
                set1.apply_previous_settings(set2)
            }
            _ => (),
        }
    }
}

impl DataUniformBuilder for SegmentData {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        match self {
            SegmentData::Scalar(_, colormap) => colormap.get_value().build_uniform(device),
            _ => None,
        }
    }

    fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform) {
        match self {
            SegmentData::Scalar(_, colormap) => {
                colormap.get_value().refresh_buffer(queue, data_uniform)
            }
            _ => (),
        }
    }
}

impl SegmentData {
    fn sphere_desc<'a>(&self) -> wgpu::VertexBufferLayout<'a> {
        match self {
            SegmentData::Color(_) => SphereColorData::desc(),
            SegmentData::Scalar(..) => SphereScalarData::desc(),
        }
    }

    fn cylinder_desc<'a>(&self) -> wgpu::VertexBufferLayout<'a> {
        match self {
            SegmentData::Color(_) => CylinderColorData::desc(),
            SegmentData::Scalar(..) => CylinderScalarData::desc(),
        }
    }

    fn build_sphere_data_buffer(&self, device: &wgpu::Device) -> wgpu::Buffer {
        match self {
            SegmentData::Scalar(scalars, _) => {
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
                    label: Some("Segment Sphere Center Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SegmentData::Color(colors) => {
                let gpu_vertices: Vec<_> = colors
                    .iter()
                    .map(|color| SphereColorData { color: *color })
                    .collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Segment Sphere Center Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }

    fn build_cylinder_data_buffer(
        &self,
        device: &wgpu::Device,
        connections: &[[u32; 2]],
    ) -> wgpu::Buffer {
        match self {
            SegmentData::Scalar(scalars, _) => {
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
                let scalars: Vec<_> = scalars
                    .iter()
                    .map(|data| (data - min_d) / (max_d - min_d))
                    .collect();
                let mut gpu_vertices = Vec::with_capacity(connections.len());
                for connection in connections {
                    let vertex = CylinderScalarData {
                        scalar_1: scalars[connection[0] as usize],
                        scalar_2: scalars[connection[1] as usize],
                    };
                    gpu_vertices.push(vertex);
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Segment Cylinder Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
            SegmentData::Color(colors) => {
                let mut gpu_vertices = Vec::with_capacity(connections.len());
                for connection in connections {
                    let vertex = CylinderColorData {
                        color_1: colors[connection[0] as usize],
                        color_2: colors[connection[1] as usize],
                    };
                    gpu_vertices.push(vertex);
                }
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Segment Cylinder Data Buffer"),
                    contents: bytemuck::cast_slice(&gpu_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                })
            }
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct PCSettings {
    radius: Radius,
    color: ColorSettings,
}

impl ShapeSettings for PCSettings {
    fn new(name: &str, l: f32) -> Self {
        let radius = Radius::new(0.15 * l);
        let color = ColorSettings::new(name);
        PCSettings { radius, color }
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui, _rebuild_pipeline: &mut bool) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= self.color.draw_ui(ui);
            changed |= self.radius.draw_ui(ui);
        });
        changed
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

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CylinderData {
    position_1: [f32; 3],
    position_2: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CylinderColorData {
    color_1: [f32; 3],
    color_2: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CylinderScalarData {
    scalar_1: f32,
    scalar_2: f32,
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

impl Vertex for CylinderData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl Vertex for CylinderColorData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl Vertex for CylinderScalarData {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<f32>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct SegmentGeometry {
    pub positions: Vec<[f32; 3]>,
    pub connections: Vec<[u32; 2]>,
    avg_edge_length: f32,
}

impl ShapeGeometry for SegmentGeometry {
    type Args = (Vec<[f32; 3]>, Vec<[u32; 2]>);

    fn new(args: Self::Args) -> Self {
        let (positions, connections) = args;
        let avg_edge_length = connections.iter().fold(0., |acc, [i0, i1]| {
            let v0 = positions[*i0 as usize];
            let v1 = positions[*i1 as usize];
            let edge = [v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]];
            acc + (edge[0].powi(2) + edge[1].powi(2) + edge[2].powi(2)).sqrt()
                / connections.len() as f32
        });
        SegmentGeometry {
            positions,
            connections,
            avg_edge_length,
        }
    }

    fn get_positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    fn get_total_elements(&self) -> u32 {
        self.positions.len() as u32 + self.connections.len() as u32
    }

    fn can_be_replaced_by(&self, other: &Self) -> bool {
        self.positions.len() == other.positions.len() && self.connections == other.connections
    }

    fn get_vertex_pos(&self, vertex: u32) -> [f32; 3] {
        self.positions[vertex as usize]
    }

    fn move_vertex(
        &mut self,
        vertex: u32,
        pos: [f32; 3],
    ) -> ((Vec<u32>, Vec<[f32; 3]>), (Vec<u32>, Vec<[f32; 3]>)) {
        self.positions[vertex as usize] = pos;
        let adj_faces = Vec::new();
        let adj_faces_centers = Vec::new();
        let mut adj_edges = Vec::with_capacity(7);
        let mut adj_edges_centers = Vec::with_capacity(7);
        for (edge_index, edge) in self.connections.iter().enumerate() {
            if edge[0] == vertex || edge[1] == vertex {
                let v0 = self.positions[edge[0] as usize];
                let v1 = self.positions[edge[1] as usize];
                adj_edges.push(edge_index as u32);
                adj_edges_centers.push([
                    (v0[0] + v1[0]) * 0.5,
                    (v0[1] + v1[1]) * 0.5,
                    (v0[2] + v1[2]) * 0.5,
                ]);
            }
        }
        (
            (adj_faces, adj_faces_centers),
            (adj_edges, adj_edges_centers),
        )
    }

    fn get_characteristic_length(&self) -> f32 {
        self.avg_edge_length
    }
}

pub struct SegmentFixedRenderer {
    positions_len: u32,
    connections_len: u32,
    vertex_buffer: wgpu::Buffer,
    center_buffer: wgpu::Buffer,
    cylinder_buffer: wgpu::Buffer,
}

pub struct SegmentDataBuffer {
    sphere_data_buffer: Option<wgpu::Buffer>,
    cylinder_data_buffer: Option<wgpu::Buffer>,
}

pub struct SegmentPipeline {
    sphere_render_pipeline: wgpu::RenderPipeline,
    cylinder_render_pipeline: wgpu::RenderPipeline,
    sphere_picker_render_pipeline: wgpu::RenderPipeline,
    cylinder_picker_render_pipeline: wgpu::RenderPipeline,
}

impl DataBuffer for SegmentDataBuffer {
    type Data = SegmentData;
    type Geometry = SegmentGeometry;

    fn new(device: &wgpu::Device, geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self {
        let sphere_data_buffer = data.map(|d| d.build_sphere_data_buffer(device));
        let cylinder_data_buffer =
            data.map(|d| d.build_cylinder_data_buffer(device, &geometry.connections));
        Self {
            sphere_data_buffer,
            cylinder_data_buffer,
        }
    }
}

impl FixedRenderer for SegmentFixedRenderer {
    type Geometry = SegmentGeometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self {
        //let s2 = 2_f32.sqrt();
        let s2 = 1.;
        let positions = [[-s2, -s2, 0.], [s2, -s2, 0.], [-s2, s2, 0.], [s2, s2, 0.]];
        let vertices = positions.map(|position| SphereVertex { position });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("PC Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let gpu_vertices = geometry
            .positions
            .iter()
            .map(|position| SphereCenter {
                position: *position,
            })
            .collect::<Vec<_>>();

        let center_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Segment Sphere Center Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let gpu_vertices2 = geometry
            .connections
            .iter()
            .map(|connection| CylinderData {
                position_1: geometry.positions[connection[0] as usize],
                position_2: geometry.positions[connection[1] as usize],
            })
            .collect::<Vec<_>>();

        let cylinder_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Segment Cylinder Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices2),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            vertex_buffer,
            center_buffer,
            cylinder_buffer,
            connections_len: geometry.connections.len() as u32,
            positions_len: geometry.positions.len() as u32,
        }
    }

    fn update_vertex(&mut self, queue: &wgpu::Queue, vertex: u32, geometry: &Self::Geometry) {
        let offset = (size_of::<SphereCenter>() * vertex as usize) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.center_buffer,
            offset,
            bytemuck::cast_slice(&[SphereCenter {
                position: geometry.positions[vertex as usize],
            }]),
        );

        for (i, connection) in geometry.connections.iter().enumerate() {
            if connection[0] == vertex || connection[1] == vertex {
                let offset = (size_of::<CylinderData>() * i as usize) as wgpu::BufferAddress;
                queue.write_buffer(
                    &self.cylinder_buffer,
                    offset,
                    bytemuck::cast_slice(&[CylinderData {
                        position_1: geometry.positions[connection[0] as usize],
                        position_2: geometry.positions[connection[1] as usize],
                    }]),
                );
            }
        }
    }
}

impl RenderPipeline for SegmentPipeline {
    type Settings = PCSettings;
    type Data = SegmentData;
    type Geometry = SegmentGeometry;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        geometry: &Self::Geometry,
        _settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let bind_group_layouts = if let Some(uniform) = data_uniform {
            vec![
                camera_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout,
                &uniform.bind_group_layout,
            ]
        } else {
            vec![
                camera_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout,
            ]
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sphere Cloud Render Pipeline Layout"),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let picker_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sphere Cloud Picker Render Pipeline Layout"),
                bind_group_layouts: &[
                    camera_bind_group_layout,
                    counter_bind_group_layout,
                    &transform_uniform.bind_group_layout,
                    &settings_uniform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("sphere cloud shader"),
            source: wgpu::ShaderSource::Wgsl(super::sphere_shader::get_shader(data).into()),
        };

        let sphere_picker_shader = include_wgsl!("../point_cloud/picker.wgsl");

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
            Some(texture::DEPTH_FORMAT),
            &sphere_buffer_layout,
            shader,
            Some("segment sphere render"),
        );

        let sphere_picker_render_pipeline = util::create_quad_picker_pipeline(
            device,
            &picker_pipeline_layout,
            texture::PICKER_FORMAT,
            Some(texture::DEPTH_FORMAT),
            &[SphereVertex::desc(), SphereCenter::desc()],
            sphere_picker_shader,
            Some("Curve Sphere picker"),
            None,
        );

        let cylinder_shader = wgpu::ShaderModuleDescriptor {
            label: Some("segment cylinder shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::get_shader(data).into()),
        };

        let cylinder_picker_shader = wgpu::ShaderModuleDescriptor {
            label: Some("segment cylinder shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::CYLINDER_PICKER_SHADER.into()),
        };
        let cylinder_buffer_layout = if let Some(data) = &data {
            vec![
                SphereVertex::desc(),
                CylinderData::desc(),
                data.cylinder_desc(),
            ]
        } else {
            vec![SphereVertex::desc(), CylinderData::desc()]
        };
        let cylinder_render_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            Some(texture::DEPTH_FORMAT),
            &cylinder_buffer_layout,
            cylinder_shader,
            Some("segment cylinder render"),
        );
        let cylinder_picker_render_pipeline = util::create_quad_picker_pipeline(
            device,
            &picker_pipeline_layout,
            texture::PICKER_FORMAT,
            Some(texture::DEPTH_FORMAT),
            &[SphereVertex::desc(), CylinderData::desc()],
            cylinder_picker_shader,
            Some("Curve Cylinder picker"),
            Some(geometry.positions.len() as u32),
        );
        SegmentPipeline {
            sphere_render_pipeline,
            cylinder_render_pipeline,
            sphere_picker_render_pipeline,
            cylinder_picker_render_pipeline,
        }
    }

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        _settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) {
        let bind_group_layouts = if let Some(uniform) = data_uniform {
            vec![
                camera_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout,
                &uniform.bind_group_layout,
            ]
        } else {
            vec![
                camera_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout,
            ]
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sphere Cloud Render Pipeline Layout"),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("sphere cloud shader"),
            source: wgpu::ShaderSource::Wgsl(super::sphere_shader::get_shader(data).into()),
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
            Some(texture::DEPTH_FORMAT),
            &sphere_buffer_layout,
            shader,
            Some("segment sphere render"),
        );

        let cylinder_shader = wgpu::ShaderModuleDescriptor {
            label: Some("segment cylinder shader"),
            source: wgpu::ShaderSource::Wgsl(super::shader::get_shader(data).into()),
        };

        let cylinder_buffer_layout = if let Some(data) = &data {
            vec![
                SphereVertex::desc(),
                CylinderData::desc(),
                data.cylinder_desc(),
            ]
        } else {
            vec![SphereVertex::desc(), CylinderData::desc()]
        };
        let cylinder_render_pipeline = util::create_quad_pipeline(
            device,
            &pipeline_layout,
            Some(texture::DEPTH_FORMAT),
            &cylinder_buffer_layout,
            cylinder_shader,
            Some("segment cylinder render"),
        );

        self.sphere_render_pipeline = sphere_render_pipeline;
        self.cylinder_render_pipeline = cylinder_render_pipeline;
    }
}

type SegmentRenderer = Renderer<SegmentFixedRenderer, SegmentDataBuffer, SegmentPipeline>;

impl SegmentRenderer {
    pub(crate) fn render_attached<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
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
        render_pass.draw(0..4, 0..(self.fixed.positions_len));

        render_pass.set_pipeline(&self.pipeline.cylinder_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.cylinder_buffer.slice(..));
        if let Some(data_buffer) = &self.data_buffer.cylinder_data_buffer {
            render_pass.set_vertex_buffer(2, data_buffer.slice(..));
        }
        render_pass.draw(0..4, 0..(self.fixed.connections_len));
    }
}

impl Render for SegmentRenderer {
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
        render_pass.draw(0..4, 0..(self.fixed.positions_len));

        render_pass.set_pipeline(&self.pipeline.cylinder_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.cylinder_buffer.slice(..));
        if let Some(data_buffer) = &self.data_buffer.cylinder_data_buffer {
            render_pass.set_vertex_buffer(2, data_buffer.slice(..));
        }
        render_pass.draw(0..4, 0..(self.fixed.connections_len));
    }

    fn render_picker<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(2, &self.transform_uniform.bind_group, &[]);
        render_pass.set_bind_group(3, &self.settings_uniform.bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline.sphere_picker_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.center_buffer.slice(..));
        render_pass.draw(0..4, 0..(self.fixed.positions_len));

        render_pass.set_pipeline(&self.pipeline.cylinder_picker_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.fixed.cylinder_buffer.slice(..));
        render_pass.draw(0..4, 0..(self.fixed.connections_len));
    }
}

pub struct SegmentDesc;

impl InvariantShapeDescriptor for SegmentDesc {
    type Data = SegmentData;
    type Geometry = SegmentGeometry;
    type Settings = PCSettings;
}

impl ShapeDescriptor<InnerBareState> for SegmentDesc {
    type Renderer = ();
    type AttachedGeometry = ();
}

impl ShapeDescriptor<InnerGraphicalState> for SegmentDesc {
    type Renderer = SegmentRenderer;
    type AttachedGeometry = EmptyAttached;
}

pub type Segment<S> = Shape<S, SegmentDesc>;
pub type UninitedSegment = Segment<InnerBareState>;
pub type DisplaySegment = Segment<InnerGraphicalState>;
pub type SegmentMut<'a, S> = ShapeMut<'a, Segment<S>, S>;

impl<S: ContextHolder> SegmentMut<'_, S>
where
    SegmentDesc: ShapeDescriptor<S>,
    Segment<S>: ShapeTrait<S, Desc = SegmentDesc>,
{
    pub fn set_radius(&mut self, radius: f32, relative: bool) -> &mut Self {
        if relative {
            self.inner.settings.radius.set_relative(radius);
        } else {
            self.inner.settings.radius.set_absolute(radius);
        }
        self.update_settings(false)
    }

    pub fn set_color(&mut self, color: [f32; 4]) -> &mut Self {
        self.inner.settings.color.color = color;
        self.update_settings(false);
        self
    }

    pub fn add_scalar(
        &mut self,
        name: impl Into<String>,
        datas: impl Scalar,
    ) -> ColorMapMut<'_, S> {
        let datas = datas.into();
        assert!(datas.len() == self.geometry().positions.len());
        let settings = ColorMap::new(&datas, S::get_settings(&self.context));
        self.add_data(name.into(), SegmentData::Scalar(datas, settings))
            .convert(|data| {
                if let SegmentData::Scalar(_, settings) = data {
                    settings
                } else {
                    panic!()
                }
            })
    }

    pub fn add_colors(&mut self, name: impl Into<String>, datas: impl Color) {
        let datas = datas.into();
        assert!(datas.len() == self.geometry().positions.len());
        self.add_data(name.into(), SegmentData::Color(datas));
    }
}

impl DisplaySegment {
    pub(crate) fn get_element(&self, index: u32) -> SegmentPicked {
        if index < self.geometry().positions.len() as u32 {
            SegmentPicked::Point(index)
        } else {
            SegmentPicked::Edge(index - self.geometry().positions.len() as u32)
        }
    }
}
