use super::data::{SurfaceData, VertexScalarSettings, VertexScalarSettingsMut};
use super::shader::{PICKER_SHADER, SHADOW_SHADER, get_shader};
use crate::attachment::{NewVectorField, VectorField, VectorFieldSettings, VectorFieldSettingsMut};
use crate::camera::Camera;
use crate::data::{internal::*, *};
use crate::geometry::*;
use crate::picker::SurfacePicked;
use crate::texture;
use crate::types::{Color, Scalar, Vertices};
use crate::types::{SurfaceIndices, Vertices2D};
use crate::ui::UiDataElement;
use crate::util;
use crate::util::Vertex;
use num_traits::cast::ToPrimitive;
use wgpu::util::DeviceExt;
use wgpu::{BufferAddress, BufferSize};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceSettingsValue {
    color: ColorSettings,
}

#[derive(Default)]
pub struct SurfaceSettings {
    color: ColorSettings,
    smooth: bool,
    show_edges: bool,
}

impl DataUniformBuilder for SurfaceSettings {
    fn build_uniform(&self, device: &wgpu::Device) -> Option<DataUniform> {
        SurfaceSettingsValue { color: self.color }.build_uniform(device)
    }

    fn refresh_buffer(&self, queue: &wgpu::Queue, data_uniform: &DataUniform) {
        SurfaceSettingsValue { color: self.color }.refresh_buffer(queue, data_uniform)
    }
}

impl NamedSettings for SurfaceSettings {
    fn set_name(mut self, name: &str) -> Self {
        self.color = ColorSettings::new(name);
        self
    }

    fn draw_ui(&mut self, ui: &mut egui::Ui, rebuild_pipeline: &mut bool) -> bool {
        //let changed = self.radius.draw(ui);
        let mut changed = false;
        ui.horizontal(|ui| {
            changed |= self.color.draw_ui(ui);
            *rebuild_pipeline |= ui.checkbox(&mut self.show_edges, "Edges").changed();
            *rebuild_pipeline |= ui.checkbox(&mut self.smooth, "Smooth").changed();
        });
        changed
    }
}
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SurfaceVertex {
    pub position: [f32; 3],
    pub normal: [i8; 4],
    //held in normal's 4th coordinate
    //pub barycentric_coords: i8,
    pub face_normal: [i8; 4],
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
                    format: wgpu::VertexFormat::Snorm8x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<([f32; 3], [i8; 4])>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Snorm8x4,
                },
            ],
        }
    }
}

pub struct SurfaceGeometry {
    pub vertices: Vec<[f32; 3]>,
    pub indices: SurfaceIndices,
    face_to_edge: FaceToEdge,
    vertex_to_face: VertexToFace,
}

struct FaceToEdge {
    indices: Vec<u32>,
    num_edges: u32,
    // same strides as indices
}

struct VertexToFace {
    indices: Vec<u32>,
    strides: Vec<u32>,
}

impl std::ops::Index<usize> for VertexToFace {
    type Output = [u32];

    fn index(&self, index: usize) -> &Self::Output {
        &self.indices
            [self.strides[index as usize] as usize..self.strides[index as usize + 1] as usize]
    }
}

fn compute_edge_face_maps(
    indices: &SurfaceIndices,
    num_vertices: usize,
) -> (FaceToEdge, VertexToFace) {
    //should compute: face_to_edge
    // vertex_to_faces
    let len = match indices {
        SurfaceIndices::Triangles(t) => 3 * t.len(),
        SurfaceIndices::Quads(q) => 4 * q.len(),
        SurfaceIndices::Polygons(i, _s) => i.len(),
    };
    let mut degrees = vec![0_u32; num_vertices + 1];
    let mut vertex_to_face_deg = vec![0_u32; num_vertices + 1];
    for face in indices {
        for i in 0..face.len() {
            vertex_to_face_deg[face[i] as usize] += 1;
            let j = if i + 1 < face.len() { i + 1 } else { 0 };
            if face[i] <= face[j] {
                degrees[face[i] as usize] += 1;
            } else {
                degrees[face[j] as usize] += 1;
            }
        }
    }
    let mut offset_1 = 0;
    let vertex_to_faces_stride: Vec<_> = vertex_to_face_deg
        .into_iter()
        .map(|v| {
            let value = v as u32;
            offset_1 += value;
            offset_1 - value
        })
        .collect();
    let mut vertex_to_face_values = vec![0_u32; offset_1 as usize];

    let mut offset = 0;
    let faces_deg: Vec<_> = degrees
        .into_iter()
        .map(|v| {
            let value = v as u32;
            offset += value;
            offset - value
        })
        .collect();

    let mut processed_by_vertex = vec![0_u32; num_vertices];
    let mut face_processed_by_vertex = vec![0_u32; num_vertices];
    let mut edges = vec![(0_u32, 0_u32); offset as usize];
    let mut face_to_edge = vec![0_u32; len];
    let mut cur_edge = 0_u32;
    let mut tot_index = 0_usize;

    let mut helper = |(v1, v2): (usize, usize)| {
        let offset = faces_deg[v1] as usize;
        let slice = &mut edges[offset..offset + processed_by_vertex[v1] as usize + 1];
        if let Some(position) = slice.iter().position(|value| value.0 == v2 as u32) {
            face_to_edge[tot_index] = slice[position].1;
        } else {
            face_to_edge[tot_index] = cur_edge;
            slice[processed_by_vertex[v1] as usize] = (v2 as u32, cur_edge);

            cur_edge += 1;
            processed_by_vertex[v1] += 1;
        }
        tot_index += 1;
    };

    for (face_index, face) in indices.into_iter().enumerate() {
        for i in 0..face.len() {
            let j = if i + 1 < face.len() { i + 1 } else { 0 };
            if face[i] <= face[j] {
                helper((face[i] as usize, face[j] as usize));
            } else {
                helper((face[j] as usize, face[i] as usize));
            }

            let offset = vertex_to_faces_stride[face[i] as usize] as usize
                + face_processed_by_vertex[face[i] as usize] as usize;
            vertex_to_face_values[offset] = face_index as u32;

            face_processed_by_vertex[face[i] as usize] += 1;
        }
    }
    (
        FaceToEdge {
            indices: face_to_edge,
            num_edges: cur_edge,
        },
        VertexToFace {
            indices: vertex_to_face_values,
            strides: vertex_to_faces_stride,
        },
    )
}

impl ElementGeometry for SurfaceGeometry {
    type Args = (SurfaceIndices, Vec<[f32; 3]>);

    fn new(args: Self::Args) -> Self {
        let (indices, vertices) = args;
        let (face_to_edge, vertex_to_face) = compute_edge_face_maps(&indices, vertices.len());
        SurfaceGeometry {
            indices,
            vertices,
            face_to_edge,
            vertex_to_face,
        }
    }

    fn get_positions(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    fn get_total_elements(&self) -> u32 {
        self.indices.tot_triangles() as u32
    }

    fn can_be_replaced_by(&self, other: &Self) -> bool {
        self.vertices.len() == other.vertices.len() && self.indices == other.indices
    }

    fn get_vertex_pos(&self, vertex: u32) -> [f32; 3] {
        self.vertices[vertex as usize]
    }

    fn move_vertex(&mut self, vertex: u32, pos: [f32; 3]) {
        self.vertices[vertex as usize] = pos;
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
    shadow_render_pipeline: wgpu::RenderPipeline,
    picker_render_pipeline: wgpu::RenderPipeline,
}

impl DataBuffer for SurfaceDataBuffer {
    type Data = SurfaceData;
    type Geometry = SurfaceGeometry;

    fn new(device: &wgpu::Device, geometry: &Self::Geometry, data: Option<&Self::Data>) -> Self {
        //let sphere_data_buffer = data.map(|d| d.build_sphere_data_buffer(device));
        let data_buffer = data.map(|d| {
            d.build_vertex_buffer(device, &geometry.indices, &geometry.face_to_edge.indices)
        });
        Self {
            data_buffer,
            // sphere_data_buffer,
        }
    }
}
fn get_barycentric_coords(j: usize, k: usize, face_len: usize) -> i8 {
    if face_len == 3 {
        match k {
            0 => 4,
            1 => 2,
            _ => 1,
        }
    } else {
        match j {
            1 => match k {
                0 => 6,
                1 => 2,
                _ => 3,
            },
            _ if j == (face_len - 2) => match k {
                0 => 5,
                1 => 3,
                _ => 1,
            },
            _ => match k {
                0 => 7,
                1 => 3,
                _ => 3,
            },
        }
    }
}

impl FixedRenderer for SurfaceFixedRenderer {
    type Geometry = SurfaceGeometry;

    fn initialize(device: &wgpu::Device, geometry: &Self::Geometry) -> Self {
        //let s2 = 2_f32.sqrt();
        let normals = compute_normals(&geometry.vertices, &geometry.indices);
        let face_normals = compute_face_normals(&geometry.vertices, &geometry.indices);
        let mut gpu_vertices = Vec::with_capacity(3 * geometry.get_total_elements() as usize);
        for (face, face_normal) in geometry.indices.into_iter().zip(face_normals) {
            for j in 1..face.len() - 1 {
                for k in 0..3 {
                    let barycentric_coords = if face.len() == 3 {
                        match k {
                            0 => 4,
                            1 => 2,
                            _ => 1,
                        }
                    } else {
                        match j {
                            1 => match k {
                                0 => 6,
                                1 => 2,
                                _ => 3,
                            },
                            _ if j == (face.len() - 2) => match k {
                                0 => 5,
                                1 => 3,
                                _ => 1,
                            },
                            _ => match k {
                                0 => 7,
                                1 => 3,
                                _ => 3,
                            },
                        }
                    };
                    let index = if k != 0 { (j - 1 + k) as usize } else { 0 };
                    let mut normal = normals[face[index] as usize];
                    normal[3] = barycentric_coords;
                    gpu_vertices.push(SurfaceVertex {
                        position: geometry.vertices[face[index] as usize],
                        normal,
                        face_normal,
                        //barycentric_coords,
                    });
                }
            }
        }
        let vertices_len = gpu_vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&gpu_vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            vertex_buffer,
            vertices_len,
        }
    }

    fn update_vertex(&mut self, queue: &wgpu::Queue, vertex: u32, geometry: &Self::Geometry) {
        let mut adj_vertices = Vec::with_capacity(7);
        let adj_faces = &geometry.vertex_to_face[vertex as usize];
        for face_index in adj_faces {
            for vertex in &geometry.indices[*face_index as usize] {
                if !adj_vertices.contains(vertex) {
                    adj_vertices.push(*vertex);
                }
            }
        }

        let mut two_ring = Vec::with_capacity(20);
        for vertex in &adj_vertices {
            for face in &geometry.vertex_to_face[*vertex as usize] {
                if !adj_faces.contains(face) && !two_ring.contains(face) {
                    two_ring.push(*face);
                }
            }
        }

        let adj_faces_normals = adj_faces
            .iter()
            .map(|face| compute_face_normal(&geometry.vertices, &geometry.indices[*face as usize]))
            .collect::<Vec<_>>();
        let two_ring_normals = two_ring
            .iter()
            .map(|face| compute_face_normal(&geometry.vertices, &geometry.indices[*face as usize]))
            .collect::<Vec<_>>();

        let adj_normals = adj_vertices
            .iter()
            .map(|vertex| {
                let mut normal = [0., 0., 0.];
                for face in &geometry.vertex_to_face[*vertex as usize] {
                    let face_normal =
                        if let Some(position) = adj_faces.iter().position(|f| f == face) {
                            adj_faces_normals[position]
                        } else {
                            let position = two_ring.iter().position(|f| f == face).unwrap();
                            two_ring_normals[position]
                        };
                    for (a, b) in face_normal.into_iter().zip(&mut normal) {
                        *b += a;
                    }
                }
                let norm =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if norm > 0. {
                    normal[0] /= norm;
                    normal[1] /= norm;
                    normal[2] /= norm;
                }
                [
                    (normal[0] * 127.).to_i8().unwrap(),
                    (normal[1] * 127.).to_i8().unwrap(),
                    (normal[2] * 127.).to_i8().unwrap(),
                    0,
                ]
            })
            .collect::<Vec<_>>();

        let two_ring_normals = two_ring_normals
            .into_iter()
            .map(|mut normal| {
                let norm =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if norm > 0. {
                    normal[0] /= norm;
                    normal[1] /= norm;
                    normal[2] /= norm;
                }
                [
                    (normal[0] * 127.).to_i8().unwrap(),
                    (normal[1] * 127.).to_i8().unwrap(),
                    (normal[2] * 127.).to_i8().unwrap(),
                    0,
                ]
            })
            .collect::<Vec<_>>();

        let adj_faces_normals = adj_faces_normals
            .into_iter()
            .map(|mut normal| {
                let norm =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if norm > 0. {
                    normal[0] /= norm;
                    normal[1] /= norm;
                    normal[2] /= norm;
                }
                [
                    (normal[0] * 127.).to_i8().unwrap(),
                    (normal[1] * 127.).to_i8().unwrap(),
                    (normal[2] * 127.).to_i8().unwrap(),
                    0,
                ]
            })
            .collect::<Vec<_>>();

        // This is the longest by far
        for (face, face_normal) in two_ring.into_iter().zip(two_ring_normals) {
            let face_number = match &geometry.indices {
                SurfaceIndices::Triangles(_) => face,
                SurfaceIndices::Quads(_) => 2 * face,
                SurfaceIndices::Polygons(_, s) => s[face as usize] - (2 * face),
            };

            let face = &geometry.indices[face as usize];
            for j in 1..face.len() - 1 {
                let vert = [face[0], face[j], face[j + 1]];
                let v0_contained = adj_vertices.iter().position(|v| *v == vert[0]);
                let v1_contained = adj_vertices.iter().position(|v| *v == vert[1]);
                let v2_contained = adj_vertices.iter().position(|v| *v == vert[2]);
                let n_contained = v0_contained.is_some() as u8
                    + v1_contained.is_some() as u8
                    + v2_contained.is_some() as u8;
                let offset = face_number as usize * 3 + (j - 1) * 3;
                match n_contained {
                    3 => {
                        let buffer = [
                            {
                                let position = v0_contained.unwrap();
                                let mut normal = adj_normals[position];
                                normal[3] = get_barycentric_coords(j, 0, face.len());
                                SurfaceVertex {
                                    position: geometry.vertices[vert[0] as usize],
                                    normal,
                                    face_normal,
                                }
                            },
                            {
                                let position = v1_contained.unwrap();
                                let mut normal = adj_normals[position];
                                normal[3] = get_barycentric_coords(j, 1, face.len());
                                SurfaceVertex {
                                    position: geometry.vertices[vert[1] as usize],
                                    normal,
                                    face_normal,
                                }
                            },
                            {
                                let position = v2_contained.unwrap();
                                let mut normal = adj_normals[position];
                                normal[3] = get_barycentric_coords(j, 2, face.len());
                                SurfaceVertex {
                                    position: geometry.vertices[vert[2] as usize],
                                    normal,
                                    face_normal,
                                }
                            },
                        ];
                        queue.write_buffer(
                            &self.vertex_buffer,
                            (offset * size_of::<SurfaceVertex>()) as BufferAddress,
                            bytemuck::cast_slice(&buffer),
                        );
                    }
                    2 => {
                        let (pos_1, pos_2, k1, k2, adj) = if v0_contained.is_none() {
                            (v1_contained.unwrap(), v2_contained.unwrap(), 1, 2, true)
                        } else if v1_contained.is_none() {
                            (v0_contained.unwrap(), v2_contained.unwrap(), 0, 2, false)
                        } else {
                            (v0_contained.unwrap(), v1_contained.unwrap(), 0, 1, true)
                        };
                        let buffer = [
                            {
                                let mut normal = adj_normals[pos_1];
                                normal[3] = get_barycentric_coords(j, k1, face.len());
                                SurfaceVertex {
                                    position: geometry.vertices[vert[k1] as usize],
                                    normal,
                                    face_normal,
                                }
                            },
                            {
                                let mut normal = adj_normals[pos_2];
                                normal[3] = get_barycentric_coords(j, k2, face.len());
                                SurfaceVertex {
                                    position: geometry.vertices[vert[k2] as usize],
                                    normal,
                                    face_normal,
                                }
                            },
                        ];
                        if adj {
                            queue.write_buffer(
                                &self.vertex_buffer,
                                ((offset + k1) * size_of::<SurfaceVertex>()) as BufferAddress,
                                bytemuck::cast_slice(&buffer),
                            );
                        } else {
                            queue.write_buffer(
                                &self.vertex_buffer,
                                ((offset) * size_of::<SurfaceVertex>()) as BufferAddress,
                                bytemuck::cast_slice(&buffer[..1]),
                            );
                            queue.write_buffer(
                                &self.vertex_buffer,
                                ((offset + 2) * size_of::<SurfaceVertex>()) as BufferAddress,
                                bytemuck::cast_slice(&buffer[1..]),
                            );
                        }
                    }
                    1 => {
                        let (pos, k) = if v0_contained.is_some() {
                            (v0_contained.unwrap(), 0)
                        } else if v1_contained.is_some() {
                            (v1_contained.unwrap(), 1)
                        } else {
                            (v2_contained.unwrap(), 2)
                        };
                        let buffer = [{
                            let mut normal = adj_normals[pos];
                            normal[3] = get_barycentric_coords(j, k, face.len());
                            SurfaceVertex {
                                position: geometry.vertices[vert[k] as usize],
                                normal,
                                face_normal,
                            }
                        }];
                        queue.write_buffer(
                            &self.vertex_buffer,
                            ((offset + k) * size_of::<SurfaceVertex>()) as BufferAddress,
                            bytemuck::cast_slice(&buffer),
                        );
                    }
                    _ => (),
                }
            }
        }

        for (face, face_normal) in adj_faces.into_iter().zip(adj_faces_normals) {
            let face_number = match &geometry.indices {
                SurfaceIndices::Triangles(_) => *face,
                SurfaceIndices::Quads(_) => 2 * *face,
                SurfaceIndices::Polygons(_, s) => s[*face as usize] - (2 * *face),
            };

            let face = &geometry.indices[*face as usize];
            let mut buffer = Vec::with_capacity(3 * (face.len() - 2));

            for j in 1..face.len() - 1 {
                let face_indices = [face[0], face[j], face[j + 1]];
                for k in 0..3 {
                    let vertex_index = face_indices[k];
                    let position = adj_vertices
                        .iter()
                        .position(|v| *v == vertex_index)
                        .unwrap();
                    let index = if k != 0 { (j - 1 + k) as usize } else { 0 };
                    let mut normal = adj_normals[position];
                    normal[3] = get_barycentric_coords(j, k, face.len());
                    buffer.push(SurfaceVertex {
                        position: geometry.vertices[face[index] as usize],
                        normal,
                        face_normal,
                    });
                }
            }
            let mut view = queue
                .write_buffer_with(
                    &self.vertex_buffer,
                    (3 * face_number as usize * size_of::<SurfaceVertex>()) as BufferAddress,
                    BufferSize::new(buffer.len() as u64 * size_of::<SurfaceVertex>() as u64)
                        .unwrap(),
                )
                .unwrap();
            view.copy_from_slice(bytemuck::cast_slice(&buffer));
        }
    }
}

impl RenderPipeline for SurfacePipeline {
    type Settings = SurfaceSettings;
    type Data = SurfaceData;
    type Geometry = SurfaceGeometry;

    fn new(
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        _geometry: &Self::Geometry,
        settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
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
                bind_group_layouts: &[
                    camera_light_bind_group_layout,
                    &transform_uniform.bind_group_layout,
                    &settings_uniform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            }),
        };
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surface Shadow Pipeline Layout"),
                bind_group_layouts: &[
                    camera_light_bind_group_layout,
                    &transform_uniform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let picker_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Surface Picker Pipeline Layout"),
                bind_group_layouts: &[
                    camera_light_bind_group_layout,
                    counter_bind_group_layout,
                    &transform_uniform.bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("Normal Shader"),
            source: wgpu::ShaderSource::Wgsl(
                get_shader(data, settings.smooth, settings.show_edges).into(),
            ),
        };
        let shadow_shader = wgpu::ShaderModuleDescriptor {
            label: Some("Surface Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADOW_SHADER.into()),
        };

        let picker_shader = wgpu::ShaderModuleDescriptor {
            label: Some("Surface Picker Shader"),
            source: wgpu::ShaderSource::Wgsl(PICKER_SHADER.into()),
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

        let shadow_render_pipeline = util::create_shadow_render_pipeline(
            device,
            &shadow_pipeline_layout,
            texture::Texture::SHADOW_FORMAT,
            None,
            &[SurfaceVertex::desc()],
            shadow_shader,
            Some("surface sphere render"),
        );

        let picker_render_pipeline = util::create_picker_pipeline(
            device,
            &picker_pipeline_layout,
            texture::Texture::PICKER_FORMAT,
            Some(texture::Texture::DEPTH_FORMAT),
            &[SurfaceVertex::desc()],
            picker_shader,
            Some("surface picker render"),
            None,
        );
        SurfacePipeline {
            surface_render_pipeline,
            shadow_render_pipeline,
            picker_render_pipeline,
        }
    }

    fn rebuild(
        &mut self,
        device: &wgpu::Device,
        data: Option<&Self::Data>,
        settings: &Self::Settings,
        transform_uniform: &DataUniform,
        settings_uniform: &DataUniform,
        data_uniform: Option<&DataUniform>,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) {
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
                bind_group_layouts: &[
                    camera_light_bind_group_layout,
                    &transform_uniform.bind_group_layout,
                    &settings_uniform.bind_group_layout,
                ],
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

        self.surface_render_pipeline = util::create_render_pipeline(
            device,
            &pipeline_layout,
            color_format,
            Some(texture::Texture::DEPTH_FORMAT),
            &buffer_layout,
            shader,
            Some("surface sphere render"),
        );
    }
}

type SurfaceRenderer = Renderer<SurfaceFixedRenderer, SurfaceDataBuffer, SurfacePipeline>;

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

    fn render_shadow<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(1, &self.transform_uniform.bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline.shadow_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.draw(0..self.fixed.vertices_len, 0..1);
    }

    fn render_picker<'a, 'b>(&'a self, render_pass: &mut wgpu::RenderPass<'b>)
    where
        'a: 'b,
    {
        render_pass.set_bind_group(2, &self.transform_uniform.bind_group, &[]);
        render_pass.set_pipeline(&self.pipeline.picker_render_pipeline);
        render_pass.set_vertex_buffer(0, self.fixed.vertex_buffer.slice(..));
        render_pass.draw(0..self.fixed.vertices_len, 0..1);
    }
}

pub type Surface<Renderer, AttachedData> =
    Element<SurfaceGeometry, Renderer, SurfaceSettings, SurfaceData, AttachedData>;

pub type UninitedSurface = Surface<(), NewVectorField>;
pub type DisplaySurface = Surface<SurfaceRenderer, VectorField>;

impl DisplaySurface {
    pub(crate) fn get_element(
        &self,
        camera: &Camera,
        item: u32,
        pos_x: f32,
        pos_y: f32,
    ) -> SurfacePicked {
        let indices = &self.geometry.indices;
        let vertices = &self.geometry.vertices;
        let (face_index, face_indices, edges) = match indices {
            SurfaceIndices::Triangles(t) => (
                item,
                t[item as usize],
                [
                    Some(self.geometry.face_to_edge.indices[item as usize * 3]),
                    Some(self.geometry.face_to_edge.indices[item as usize * 3 + 1]),
                    Some(self.geometry.face_to_edge.indices[item as usize * 3 + 2]),
                ],
            ),
            SurfaceIndices::Quads(t) => (
                item / 2,
                if item % 2 == 0 {
                    [
                        t[item as usize / 2][0],
                        t[item as usize / 2][1],
                        t[item as usize / 2][2],
                    ]
                } else {
                    [
                        t[item as usize / 2][0],
                        t[item as usize / 2][2],
                        t[item as usize / 2][3],
                    ]
                },
                if item % 2 == 0 {
                    [
                        Some(
                            self.geometry.face_to_edge.indices
                                [(item as usize - (item % 2) as usize) * 2],
                        ),
                        Some(
                            self.geometry.face_to_edge.indices
                                [(item as usize - (item % 2) as usize) * 2 + 1],
                        ),
                        None,
                    ]
                } else {
                    [
                        None,
                        Some(
                            self.geometry.face_to_edge.indices
                                [(item as usize - (item % 2) as usize) * 2 + 2],
                        ),
                        Some(
                            self.geometry.face_to_edge.indices
                                [(item as usize - (item % 2) as usize) * 2 + 3],
                        ),
                    ]
                },
            ),
            SurfaceIndices::Polygons(indices, s) => {
                let mut elapsed = 0;
                let mut index = 0;
                let mut face = [0, 0, 0];
                let mut edges = [None, None, None];
                for (i, bounds) in s.windows(2).enumerate() {
                    let size = bounds[1] - bounds[0];
                    if elapsed + size - 2 > item {
                        for j in 0..(size - 2) {
                            if elapsed + j == item {
                                index = i as u32;
                                face = [
                                    indices[elapsed as usize + i * 2 + 0],
                                    indices[elapsed as usize + i * 2 + j as usize + 1],
                                    indices[elapsed as usize + i * 2 + j as usize + 2],
                                ];
                                if j == 0 {
                                    edges[0] = Some(
                                        self.geometry.face_to_edge.indices
                                            [elapsed as usize + i * 2 + 0],
                                    );
                                    edges[1] = Some(
                                        self.geometry.face_to_edge.indices
                                            [elapsed as usize + i * 2 + 1],
                                    );
                                } else if j == (size - 3) {
                                    edges[1] = Some(
                                        self.geometry.face_to_edge.indices
                                            [elapsed as usize + size as usize - 2],
                                    );
                                    edges[2] = Some(
                                        self.geometry.face_to_edge.indices
                                            [elapsed as usize + size as usize - 1],
                                    );
                                } else {
                                    edges[1] = Some(
                                        self.geometry.face_to_edge.indices
                                            [elapsed as usize * 2 + j as usize + 1],
                                    );
                                }
                                break;
                            }
                        }
                        break;
                    } else {
                        elapsed += size - 2;
                    }
                }
                (index, face, edges)
            }
        };
        let v1 = glam::Vec3::from_array(vertices[face_indices[0] as usize]);
        let v2 = glam::Vec3::from_array(vertices[face_indices[1] as usize]);
        let v3 = glam::Vec3::from_array(vertices[face_indices[2] as usize]);
        let v1 = v1.extend(1.);
        let v2 = v2.extend(1.);
        let v3 = v3.extend(1.);
        let model = glam::Mat4::from_cols_array_2d(&self.transform.to_raw().get_model());
        let camera = camera.build_view_projection_matrix();
        let v1 = camera * model * v1;
        let v2 = camera * model * v2;
        let v3 = camera * model * v3;
        let w1 = v1.w;
        let w2 = v2.w;
        let w3 = v3.w;
        let v1 = v1 / v1.w;
        let v2 = v2 / v2.w;
        let v3 = v3 / v3.w;
        let p = glam::Vec3::new(pos_x, pos_y, 0.);
        let v1 = glam::Vec3::new(v1.x, v1.y, 0.);
        let v2 = glam::Vec3::new(v2.x, v2.y, 0.);
        let v3 = glam::Vec3::new(v3.x, v3.y, 0.);

        let c3 = (v1 - p).cross(v2 - p).length();
        let c1 = (v2 - p).cross(v3 - p).length();
        let c2 = (v3 - p).cross(v1 - p).length();
        let c1p = c1 / w1 / (c1 / w1 + c2 / w2 + c3 / w3);
        let c2p = c2 / w2 / (c1 / w1 + c2 / w2 + c3 / w3);
        let c3p = c3 / w3 / (c1 / w1 + c2 / w2 + c3 / w3);
        if c1p > 0.7 {
            SurfacePicked::Vertex(face_indices[0])
        } else if c2p > 0.7 {
            SurfacePicked::Vertex(face_indices[1])
        } else if c3p > 0.7 {
            SurfacePicked::Vertex(face_indices[2])
        } else if c1p < 0.15 && edges[1].is_some() {
            SurfacePicked::Edge(edges[1].unwrap())
        } else if c2p < 0.15 && edges[2].is_some() {
            SurfacePicked::Edge(edges[2].unwrap())
        } else if c3p < 0.15 && edges[0].is_some() {
            SurfacePicked::Edge(edges[0].unwrap())
        } else {
            SurfacePicked::Face(face_index)
        }
    }
}

pub type SurfaceMut<'a, Renderer, AttachedData, Context> =
    ElementMut<'a, Surface<Renderer, AttachedData>, Context>;

impl<'a, Renderer, AttachedData, Ctxt: Context> SurfaceMut<'a, Renderer, AttachedData, Ctxt>
where
    AttachedData: AttachedGeometry<
            Ctxt,
            Settings = VectorFieldSettings,
            Args = (Vec<[f32; 3]>, Vec<[f32; 3]>),
        >,
    Surface<Renderer, AttachedData>:
        ElementTrait<Ctxt, Data = SurfaceData, Attached = AttachedData>,
{
    pub fn show_edges(&mut self, show_edges: bool) -> &mut Self {
        if self.element.settings.show_edges != show_edges {
            self.element.settings.show_edges = show_edges;
            self.update_settings(true);
        }
        self
    }

    pub fn set_smooth(&mut self, smooth: bool) -> &mut Self {
        if self.element.settings.smooth != smooth {
            self.element.settings.smooth = smooth;
            self.update_settings(true);
        }
        self
    }

    pub fn add_face_scalar<S: Scalar>(&mut self, name: String, datas: S) -> ColorMapMut<'_, Ctxt> {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.indices.size());
        let new_settings = ColorMap::new(&datas, self.context.get_settings());
        self.add_data(name, SurfaceData::FaceScalar(datas, new_settings))
            .convert(|data| {
                if let SurfaceData::FaceScalar(_, settings) = data {
                    settings
                } else {
                    panic!()
                }
            })
    }

    pub fn add_edge_scalar<S: Scalar>(&mut self, name: String, datas: S) -> ColorMapMut<'_, Ctxt> {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.face_to_edge.num_edges as usize);
        let new_settings = ColorMap::new(&datas, self.context.get_settings());
        self.add_data(name, SurfaceData::EdgeScalar(datas, new_settings))
            .convert(|data| {
                if let SurfaceData::EdgeScalar(_, settings) = data {
                    settings
                } else {
                    panic!()
                }
            })
    }

    pub fn add_vertex_scalar<S: Scalar>(
        &mut self,
        name: String,
        datas: S,
    ) -> VertexScalarSettingsMut<'_, Ctxt> {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.vertices.len());
        let new_settings = VertexScalarSettings::new(&datas, self.context.get_settings());
        self.add_data(name, SurfaceData::VertexScalar(datas, new_settings))
            .convert(|data| {
                if let SurfaceData::VertexScalar(_, settings) = data {
                    settings
                } else {
                    panic!()
                }
            })
    }

    pub fn add_uv_map<UV: Vertices2D>(
        &mut self,
        name: String,
        datas: UV,
    ) -> UVMapSettingsMut<'_, Ctxt> {
        let datas = datas.into();
        assert!(datas.len() == self.geometry.vertices.len());
        self.add_data(name, SurfaceData::UVMap(datas, UVMapSettings::default()))
            .convert(|data| {
                if let SurfaceData::UVMap(_, settings) = data {
                    settings
                } else {
                    panic!()
                }
            })
    }

    pub fn add_corner_uv_map<UV: Vertices2D>(
        &mut self,
        name: String,
        datas: UV,
    ) -> UVMapSettingsMut<'_, Ctxt> {
        let datas = datas.into();
        assert!(datas.len() == 3 * self.geometry.indices.size());
        self.add_data(
            name,
            SurfaceData::UVCornerMap(datas, UVMapSettings::default()),
        )
        .convert(|data| {
            if let SurfaceData::UVCornerMap(_, settings) = data {
                settings
            } else {
                panic!()
            }
        })
    }

    pub fn add_vertex_color<C: Color>(&mut self, name: String, colors: C) {
        let colors = colors.into();
        assert!(colors.len() == self.geometry.vertices.len());
        self.add_data(name, SurfaceData::Color(colors));
    }

    pub fn add_vertex_vector_field<V: Vertices>(
        &mut self,
        name: String,
        vectors: V,
    ) -> VectorFieldSettingsMut<'_, Ctxt> {
        let vectors = vectors.into();
        assert!(vectors.len() == self.geometry.vertices.len());
        let offsets: Vec<[f32; 3]> = self.geometry.vertices.clone();
        self.add_attached_geometry(name.clone(), (offsets, vectors).into())
            .convert(|attached| attached.get_settings())
    }

    pub fn add_face_vector_field<V: Vertices>(
        &mut self,
        name: String,
        vectors: V,
    ) -> VectorFieldSettingsMut<'_, Ctxt> {
        let vectors = vectors.into();
        assert!(vectors.len() == self.geometry.indices.size());
        let offsets: Vec<[f32; 3]> = self
            .geometry
            .indices
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

        self.add_attached_geometry(name.clone(), (offsets, vectors).into())
            .convert(|attached| attached.get_settings())
    }
}

//Can be simplified now using vertex -> face adjacency
fn compute_normals(vertices: &[[f32; 3]], indices: &SurfaceIndices) -> Vec<[i8; 4]> {
    let mut normals = vec![[0., 0., 0.]; vertices.len()];
    for face in indices {
        for i in 1..face.len() - 1 {
            let i0 = face[0] as usize;
            let i1 = face[i] as usize;
            let i2 = face[i + 1] as usize;
            let v0 = glam::Vec3::from_array(vertices[i0]);
            let v1 = glam::Vec3::from_array(vertices[i1]);
            let v2 = glam::Vec3::from_array(vertices[i2]);
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
        let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if norm > 0. {
            normal[0] /= norm;
            normal[1] /= norm;
            normal[2] /= norm;
        }
    }
    normals
        .into_iter()
        .map(|n| {
            [
                (n[0] * 127.).to_i8().unwrap(),
                (n[1] * 127.).to_i8().unwrap(),
                (n[2] * 127.).to_i8().unwrap(),
                0,
            ]
        })
        .collect()
}

fn compute_face_normal(vertices: &[[f32; 3]], face: &[u32]) -> [f32; 3] {
    let mut normal = [0., 0., 0.];
    for i in 1..face.len() - 1 {
        let i0 = face[0] as usize;
        let i1 = face[i] as usize;
        let i2 = face[i + 1] as usize;
        let v0 = glam::Vec3::from_array(vertices[i0]);
        let v1 = glam::Vec3::from_array(vertices[i1]);
        let v2 = glam::Vec3::from_array(vertices[i2]);
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let cross_p = e1.cross(e2);
        let n = AsRef::<[f32; 3]>::as_ref(&cross_p);
        for (a, b) in normal.iter_mut().zip(n) {
            *a += b
        }
    }
    normal
}

fn compute_face_normals(vertices: &[[f32; 3]], indices: &SurfaceIndices) -> Vec<[i8; 4]> {
    indices
        .into_iter()
        .map(|face| {
            let mut normal = compute_face_normal(vertices, face);

            let norm =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if norm > 0. {
                normal[0] /= norm;
                normal[1] /= norm;
                normal[2] /= norm;
            }
            [
                (normal[0] * 127.).to_i8().unwrap(),
                (normal[1] * 127.).to_i8().unwrap(),
                (normal[2] * 127.).to_i8().unwrap(),
                0,
            ]
        })
        .collect()
}
