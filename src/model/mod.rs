use std::collections::HashMap;

use crate::model::data::{DataUniform, MeshData};
use crate::model::mesh_picker::MeshPicker;
use crate::model::renderer::{MeshRenderer, ModelVertex};
use crate::texture;

mod arrow_shader;
pub mod data;
mod mesh_picker;
mod mesh_shader;
mod renderer;
pub mod vector_field;

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub bind_group: wgpu::BindGroup,
}

pub struct Mesh {
    pub name: String,
    pub vertices: Vec<ModelVertex>,
    //pub normals: Vec<[f32; 3]>,
    pub indices: Vec<[u32; 3]>,
    pub num_elements: u32,
    pub material: usize,
    pub transform: Transform,
    pub property_changed: bool,
    pub transform_changed: bool,
    pub uniform_changed: bool,
    pub datas: HashMap<String, MeshData>,
    pub shown_data: Option<String>,
    // When the data to show is changed, so we can do the change in the main loop
    pub color: [f32; 4],
    pub show: bool,
    pub show_edges: bool,
    pub smooth: bool,
    pub show_gizmo: bool,
    pub gizmo_mode: egui_gizmo::GizmoMode,
    pub data_to_show: Option<Option<String>>,
}

//TODO refactor datas properties
/*
struct MeshProperties {
    pub show_edges: bool,
    pub smooth: bool,
    pub shown_data: Option<String>,
}
*/

pub struct Transform(pub [[f32; 4]; 4]);

impl Default for Transform {
    fn default() -> Self {
        Transform([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    }
}

impl Transform {
    pub fn to_raw(&self) -> TransformRaw {
        use cgmath::Matrix;
        use cgmath::SquareMatrix;
        let model = self.0;
        let mut normal: [[f32; 3]; 3] = [[0.; 3]; 3];
        for (row, row_orig) in normal.iter_mut().zip(model) {
            *row = row_orig[0..3].try_into().unwrap();
        }
        let mut normal: cgmath::Matrix3<f32> = normal.into();
        normal = normal.invert().unwrap().transpose();
        //Conversion tricks from mat3x3 to mat4x4
        let normal: cgmath::Matrix4<f32> = normal.into();
        TransformRaw {
            model,
            normal: normal.into(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TransformRaw {
    model: [[f32; 4]; 4],
    //Actually 3x3 but mat4 for alignment
    normal: [[f32; 4]; 4],
}

impl TransformRaw {
    pub fn get_model(&self) -> [[f32; 4]; 4] {
        self.model
    }
}

impl Mesh {
    pub fn new(name: &str, vertices: &Vec<[f32; 3]>, indices: &Vec<[u32; 3]>) -> Self {
        let normals = Self::compute_normals(vertices, indices);
        let vertices = vertices
            .iter()
            .zip(normals)
            .map(|(vertex, normal)| ModelVertex {
                position: [vertex[0], vertex[1], vertex[2]],
                tex_coords: [0., 0.],
                normal,
                color: [0.2, 0.2, 0.8],
                barycentric_coords: [0., 0., 0.],
                distance: 0.,
            })
            .collect::<Vec<_>>();
        let indices = indices.clone();
        let transform = Transform::default();
        let num_elements = indices.len() as u32;
        Self {
            name: name.to_string(),
            vertices,
            indices,
            num_elements,
            material: 0,
            transform,
            transform_changed: false,
            uniform_changed: false,
            datas: std::collections::HashMap::new(),
            shown_data: None,
            show: true,
            color: [0.2, 0.2, 0.8, 1.],
            show_edges: false,
            smooth: true,
            data_to_show: None,
            show_gizmo: false,
            gizmo_mode: egui_gizmo::GizmoMode::Translate,
            property_changed: false,
        }
    }

    fn compute_normals(vertices: &Vec<[f32; 3]>, indices: &Vec<[u32; 3]>) -> Vec<[f32; 3]> {
        let mut normals = vec![[0., 0., 0.]; vertices.len()];
        for face in indices {
            let i0 = face[0] as usize;
            let i1 = face[1] as usize;
            let i2 = face[2] as usize;
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
        for normal in &mut normals {
            let norm =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            normal[0] /= norm;
            normal[1] /= norm;
            normal[2] /= norm;
        }
        normals
    }

    pub fn show_edges(&mut self, show_edges: bool) -> &mut Self {
        if self.show_edges != show_edges {
            self.show_edges = show_edges;
            self.property_changed = true;
        }
        self
    }

    pub fn set_smooth(&mut self, smooth: bool) -> &mut Self {
        if self.smooth != smooth {
            self.smooth = smooth;
            self.property_changed = true;
        }
        self
    }

    //TODO some kind of error handling
    pub fn add_face_scalar(&mut self, name: String, datas: Vec<f32>) -> &mut Self {
        assert!(datas.len() == self.indices.len());
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        self.datas.insert(name, MeshData::FaceScalar(datas));
        self
    }

    pub fn add_vertex_scalar(&mut self, name: String, datas: Vec<f32>) -> &mut Self {
        assert!(datas.len() == self.vertices.len());
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        self.datas.insert(
            name,
            MeshData::VertexScalar(datas, crate::model::data::VertexScalarUniform::default()),
        );
        self
    }

    pub fn add_uv_map(&mut self, name: String, datas: Vec<(f32, f32)>) -> &mut Self {
        assert!(datas.len() == self.vertices.len());
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        self.datas.insert(
            name,
            MeshData::UVMap(datas, crate::model::data::UVMapUniform::default()),
        );
        self
    }

    pub fn add_corner_uv_map(&mut self, name: String, datas: Vec<(f32, f32)>) -> &mut Self {
        assert!(datas.len() == 3 * self.indices.len());
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        self.datas.insert(
            name,
            MeshData::UVCornerMap(datas, crate::model::data::UVMapUniform::default()),
        );
        self
    }

    pub fn add_vertex_color(&mut self, name: String, colors: Vec<[f32; 3]>) -> &mut Self {
        assert!(colors.len() == self.vertices.len());
        if let Some(data_name) = &mut self.shown_data {
            if *data_name == name {
                self.data_to_show = Some(Some(data_name.clone()));
                self.shown_data = None;
            }
        }
        self.datas.insert(name, MeshData::Color(colors));
        self
    }

    pub fn set_data(&mut self, name: Option<String>) -> &mut Self {
        self.data_to_show = Some(name);
        self
    }

    pub fn refresh_transform(&mut self) {
        self.transform_changed = true;
    }
}

pub struct Model {
    pub mesh: Mesh,
    renderer: MeshRenderer,
    picker: MeshPicker,
}

impl Model {
    pub fn new(
        name: &str,
        vertices: &Vec<[f32; 3]>,
        indices: &Vec<[u32; 3]>,
        device: &wgpu::Device,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        counter_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let mesh = Mesh::new(name, vertices, indices);
        let renderer =
            MeshRenderer::new(device, camera_light_bind_group_layout, color_format, &mesh);
        let picker = MeshPicker::new(
            device,
            camera_light_bind_group_layout,
            counter_bind_group_layout,
            &mesh,
        );
        Self {
            mesh,
            renderer,
            picker,
        }
    }

    pub fn refresh_data(
        &mut self,
        device: &wgpu::Device,
        queue: &mut wgpu::Queue,
        camera_light_bind_group_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> bool {
        self.picker.update(queue, &self.mesh);
        self.renderer.update(
            device,
            queue,
            camera_light_bind_group_layout,
            color_format,
            &mut self.mesh,
        )
    }

    pub fn get_total_elements(&self) -> u32 {
        self.picker.get_total_elements()
    }
}

pub trait DrawModel<'a> {
    fn draw_model(&mut self, model: &'a Model);
    fn draw_picker(&mut self, model: &'a Model);
}
impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_model(&mut self, model: &'b Model) {
        if model.mesh.show {
            model.renderer.render(self);
        }
    }

    fn draw_picker(&mut self, model: &'b Model) {
        if model.mesh.show {
            model.picker.render(self);
        }
    }
}
