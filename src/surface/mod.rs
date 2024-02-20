use crate::data::*;
use crate::attachment::{NewVectorField, VectorFieldSettings};
use crate::texture;
use crate::types::{Color, Scalar, Vertices};
use crate::updater::*;
use crate::util;
use crate::types::{Vertices2D, SurfaceIndices};
use wgpu::util::DeviceExt;
use crate::ui::UiDataElement;

mod data;
mod shader;
mod picker;
use data::*;
use shader::get_shader;
use picker::Picker;

pub trait Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a>;
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceSettingsValue {
    color: ColorSettings,
}

#[derive(Default)]
pub struct SurfaceSettings {
    pub color: ColorSettings,
    pub smooth: bool,
    pub show_edges: bool,
}

impl DataUniformBuilder for SurfaceSettings {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        SurfaceSettingsValue {
            color: self.color,
        }.build_uniform(device)
    }

    fn refresh_buffer(&self, queue: &mut wgpu::Queue, data_uniform: &DataUniform) {
        SurfaceSettingsValue {
            color: self.color,
        }.refresh_buffer(queue, data_uniform)
    }
}

impl UiDataElement for SurfaceSettings {
    fn draw(&mut self, ui: &mut egui::Ui, property_changed: &mut bool) -> bool {
        //let changed = self.radius.draw(ui);
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= self.color.draw(ui, property_changed);
            *property_changed |= ui.checkbox(&mut self.show_edges, "Edges").changed();
            *property_changed |= ui.checkbox(&mut self.smooth, "Smooth").changed();
        });
//        self.transform.draw(ui, property_changed);
        changed
    }

    fn draw_gizmo(&mut self,
        ui: &mut egui::Ui,
        name: &str,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) -> bool {
//        self.transform.draw_gizmo(ui, name, view, proj)
        false
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SurfaceVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub face_normal: [f32; 3],
    pub barycentric_coords: [f32; 3],
}

impl Vertex for SurfaceVertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 9]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

pub struct SurfaceGeometry {
    pub vertices: Vec<[f32; 3]>,
    //pub normals: Vec<[f32; 3]>,
    pub indices: SurfaceIndices,
    pub num_elements: u32,
    internal_indices: Vec<[u32; 3]>,
}

impl Positions for SurfaceGeometry {
    fn get_positions(&self) -> &[[f32; 3]] {
        &self.vertices
    }
}

pub struct SurfaceFixedRenderer {
    vertex_buffer: wgpu::Buffer,
    vertices_len: u32,
}

pub struct SurfaceDataBuffer {
    data_buffer: Option<wgpu::Buffer>,
}

pub struct SurfacePipeline {
    surface_render_pipeline: wgpu::RenderPipeline,
}

impl DataBuffer for SurfaceDataBuffer {
    type Settings = SurfaceSettings;
    type Data = SurfaceData;
    type Geometry = SurfaceGeometry;

    fn new(device: &wgpu::Device, geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self {
        //let sphere_data_buffer = data.map(|d| d.build_sphere_data_buffer(device));
        let data_buffer = data.map(|d| d.build_vertex_buffer(device, &geometry.indices, &geometry.internal_indices));
        Self {
            data_buffer,
           // sphere_data_buffer,
        }
    }
}

impl FixedRenderer for SurfaceFixedRenderer {
    type Settings = SurfaceSettings;
    type Data = SurfaceData;
    type Geometry = SurfaceGeometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self {
        //let s2 = 2_f32.sqrt();
        let normals = compute_normals(&geometry.vertices, &geometry.indices);
        let face_normals = compute_face_normals(&geometry.vertices, &geometry.indices);
        let mut gpu_vertices = Vec::with_capacity(3 * geometry.internal_indices.len());
        for (face, face_normal) in geometry.indices.into_iter().zip(face_normals) {
            let fmin2 = face.len() - 2;
            for j in 1..face.len() - 1 {
                for k in 0..3 {
                    let barycentric_coords = if face.len() == 3 {
                        match k {
                            0 => [1., 0., 0.],
                            1 => [0., 1., 0.],
                            _ => [0., 0., 1.],
                        }
                    } else {
                        match j {
                            1 => {
                                match k {
                                    0 => [1., 1., 0.],
                                    1 => [0., 1.1, 0.],
                                    _ => [0., 1., 1.],
                                }
                            },
                            fmin2 => {
                                match k {
                                    0 => [1., 0., 1.],
                                    1 => [0., 1., 1.],
                                    _ => [0., 0., 1.1],
                                }
                            },
                            _ => {
                                match k {
                                    0 => [1., 1., 1.],
                                    1 => [0., 1.1, 1.],
                                    _ => [0., 1., 1.1],
                                }
                            }
                        }
                    };
                    let index = if k != 0 { (j - 1 + k) as usize } else { 0 };
                    gpu_vertices.push(SurfaceVertex {
                        position: geometry.vertices[face[index] as usize],
                        normal: normals[face[index] as usize],
                        face_normal,
                        barycentric_coords,
                    });
                }
            }
        }
        let vertices_len = gpu_vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            //normals,
            //face_normals,
            vertex_buffer,
            vertices_len,
        }
    }
}

impl RenderPipeline for SurfacePipeline {
    type Settings = SurfaceSettings;
    type Data = SurfaceData;
    type Geometry = SurfaceGeometry;
    type Fixed = SurfaceFixedRenderer;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        fixed: &Self::Fixed,
        settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline_layout = match data_uniform {
            Some(uniform) => device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    camera_light_bind_group_layout,
                    &transform_uniform.bind_group_layout,
                    &settings_uniform.bind_group_layout,
                    &uniform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            }),
            None => device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[camera_light_bind_group_layout,
                &transform_uniform.bind_group_layout,
                &settings_uniform.bind_group_layout],
                push_constant_ranges: &[],
            }),
        };
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Normal Shader"),
            source: wgpu::ShaderSource::Wgsl(
                get_shader(data, settings.smooth, settings.show_edges).into(),
            ),
        };

        let buffer_layout = match data {
            Some(data) => vec![SurfaceVertex::desc(), data.desc()],
            None => vec![SurfaceVertex::desc()],
        };

        let surface_render_pipeline = util::create_render_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &buffer_layout,
            shader,
            Some("surface sphere render"),
        );
        SurfacePipeline {
            surface_render_pipeline,
        }
    }
}

type SurfaceRenderer =
    Renderer<SurfaceSettings, SurfaceData, SurfaceGeometry, SurfaceFixedRenderer, SurfaceDataBuffer, SurfacePipeline>;

impl Render for SurfaceRenderer {
    fn render<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.transform_uniform.bind_group, &[]);
        render_pass.set_bind_group(2, &self.settings_uniform.bind_group, &[]);
        if let Some(data_uniform) = &self.data_uniform {
            render_pass.set_bind_group(3, &data_uniform.bind_group, &[]);
        }
        render_pass.set_pipeline(&self.pipeline.surface_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        if let Some(buffer) = &self.data_buffer.data_buffer {
            render_pass.set_vertex_buffer(1, buffer.slice(..));
        }
        render_pass.draw(0..self.fixed.vertices_len, 0..1);
    }
}

pub type Surface = MainDisplayGeometry<
    SurfaceSettings,
    SurfaceData,
    SurfaceGeometry,
    SurfaceFixedRenderer,
    SurfaceDataBuffer,
    SurfacePipeline,
    Picker,
>;

impl Surface {
    pub fn new(
        name: String,
        vertices: Vec<[f32; 3]>,
        indices: SurfaceIndices,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        //transform_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let mut internal_indices = Vec::new();
        for face in &indices {
            for i in 1..face.len() - 1 {
                internal_indices.push([face[0], face[i], face[i + 1]]);
            }
        }
        let geometry = SurfaceGeometry {
            num_elements: indices.size() as u32,
            indices,
            vertices,
            internal_indices,
        };
        Surface::init(device, name, geometry, camera_light_bind_group_layout, counter_bind_group_layout, color_format)
    }

    pub fn show_edges(&mut self, show_edges: bool) -> &mut Self {
        if self.updater.settings.show_edges != show_edges {
            self.updater.settings.show_edges = show_edges;
            self.updater.property_changed = true;
        }
        self
    }

    pub fn set_smooth(&mut self, smooth: bool) -> &mut Self {
        if self.updater.settings.smooth != smooth {
            self.updater.settings.smooth = smooth;
            self.updater.property_changed = true;
        }
        self
    }

    pub fn add_face_scalar<S: Scalar>(&mut self, name: String, datas: S) -> &mut FaceScalarSettings {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.indices.size());
        if let SurfaceData::FaceScalar(_, settings) = self.updater.add_data(name, SurfaceData::FaceScalar(datas, FaceScalarSettings::default())) {
            settings
        } else {
            panic!()
        }
    }

    pub fn add_vertex_scalar<S: Scalar>(&mut self, name: String, datas: S) -> &mut VertexScalarSettings {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.vertices.len());
        if let SurfaceData::VertexScalar(_, settings) = self.updater.add_data(name, SurfaceData::VertexScalar(datas, VertexScalarSettings::default())) {
            settings
        } else {
            panic!()
        }
    }

    pub fn add_uv_map<UV: Vertices2D>(&mut self, name: String, datas: UV) -> &mut UVMapSettings {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.vertices.len());
        if let SurfaceData::UVMap(_, settings) = self.updater.add_data(name, SurfaceData::UVMap(datas, UVMapSettings::default())) {
            settings
        } else {
            panic!()
        }
    }

    pub fn add_corner_uv_map<UV: Vertices2D>(&mut self, name: String, datas: UV) -> &mut UVMapSettings {
        let datas = datas.into();
        assert!(datas.len() == 3 * self.geometry.indices.size());
        if let SurfaceData::UVCornerMap(_, settings) = self.updater.add_data(name, SurfaceData::UVCornerMap(datas, UVMapSettings::default())) {
            settings
        } else {
            panic!()
        }
    }

    pub fn add_vertex_color<C: Color>(&mut self, name: String, colors: C) -> &mut SurfaceData {
        let colors = colors.into();
        assert!(colors.len() == self.geometry.vertices.len());
        self.updater.add_data(name, SurfaceData::Color(colors))
    }

    pub fn add_vertex_vector_field<V: Vertices>(
        &mut self,
        name: String,
        vectors: V,
    ) -> &mut VectorFieldSettings {
        let vectors = vectors.into();
        assert!(vectors.len() == self.geometry.vertices.len());
        let offsets: Vec<[f32; 3]> = self.geometry.vertices.clone();
        let vector_field = NewVectorField::new(name, vectors, offsets);
        self.updater.queued_attached_data.push(vector_field);
        &mut self.updater.queued_attached_data.last_mut().unwrap().settings
    }

    pub fn add_face_vector_field<V: Vertices>(
        &mut self,
        name: String,
        vectors: V,
    ) -> &mut VectorFieldSettings {
        let vectors = vectors.into();
        assert!(vectors.len() == self.geometry.indices.size());
        let offsets: Vec<[f32; 3]> = self
            .geometry.indices
            .into_iter()
            .map(|face| {
                let mut res0 = 0.;
                let mut res1 = 0.;
                let mut res2 = 0.;
                for index in face {
                    let vertex = self.geometry.vertices[*index as usize];
                    res0 += vertex[0];
                    res1 += vertex[1];
                    res2 += vertex[2];
                }
                res0 = res0 / face.len() as f32;
                res1 = res1 / face.len() as f32;
                res2 = res2 / face.len() as f32;
                [res0, res1, res2]
            })
            .collect();
        let vector_field = NewVectorField::new(name, vectors, offsets);
        self.updater.queued_attached_data.push(vector_field);
        &mut self.updater.queued_attached_data.last_mut().unwrap().settings
    }

    pub(crate) fn draw_element_info(&self, element: usize, ui: &mut egui::Ui) {
        if element < self.geometry.vertices.len() {
            ui.label(format!("Picked vertex number {}", element));
        } else if element < self.geometry.vertices.len() + self.geometry.indices.size() {
            ui.label(format!("Picked face number {}", element - self.geometry.vertices.len()));
        }
    }
}

fn compute_normals(vertices: &[[f32; 3]], indices: &SurfaceIndices) -> Vec<[f32; 3]> {
    let mut normals = vec![[0., 0., 0.]; vertices.len()];
    for face in indices {
        for i in 1..face.len() - 1 {
            let i0 = face[0] as usize;
            let i1 = face[i] as usize;
            let i2 = face[i + 1] as usize;
            let v0: cgmath::Vector3<f32> = vertices[i0].into();
            let v1: cgmath::Vector3<f32> = vertices[i1].into();
            let v2: cgmath::Vector3<f32> = vertices[i2].into();
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let cross_p = e1.cross(e2);
            let n = AsRef::<[f32; 3]>::as_ref(&cross_p);
            for (a, b) in normals[i0].iter_mut().zip(n) {
                *a += b
            }
            for (a, b) in normals[i1].iter_mut().zip(n) {
                *a += b
            }
            for (a, b) in normals[i2].iter_mut().zip(n) {
                *a += b
            }
        }
    }
    for normal in &mut normals {
        let norm =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        normal[0] /= norm;
        normal[1] /= norm;
        normal[2] /= norm;
    }
    normals
}

fn compute_face_normals(vertices: &[[f32; 3]], indices: &SurfaceIndices) -> Vec<[f32; 3]> {
    let mut normals = vec![[0., 0., 0.]; indices.size()];
    for (normal, face) in normals.iter_mut().zip(indices) {
        for i in 1..face.len() - 1 {
            let i0 = face[0] as usize;
            let i1 = face[i] as usize;
            let i2 = face[i + 1] as usize;
            let v0: cgmath::Vector3<f32> = vertices[i0].into();
            let v1: cgmath::Vector3<f32> = vertices[i1].into();
            let v2: cgmath::Vector3<f32> = vertices[i2].into();
            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let cross_p = e1.cross(e2);
            let n = AsRef::<[f32; 3]>::as_ref(&cross_p);
            for (a, b) in normal.iter_mut().zip(n) {
                *a += b
            }
        }
    }
    for normal in &mut normals {
        let norm =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        normal[0] /= norm;
        normal[1] /= norm;
        normal[2] /= norm;
    }
    normals
}
