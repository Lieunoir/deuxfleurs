use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use crate::model::data::{DataUniform, MeshData};
use crate::model::mesh_picker::MeshPicker;
use crate::model::renderer::{MeshRenderer, ModelVertex};
use crate::model::vector_field::{VectorField, VectorFieldSettings};
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

pub struct VectorFieldData {
    pub field: VectorField,
    pub shown: bool,
}

pub struct Mesh {
    pub name: String,
    pub vertices: Vec<[f32; 3]>,
    internal_vertices: Vec<ModelVertex>,
    //pub normals: Vec<[f32; 3]>,
    //TODO use face strides
    pub indices: Vec<Vec<u32>>,
    internal_indices: Vec<[u32; 3]>,
    face_normals: Vec<[f32; 3]>,
    pub num_elements: u32,
    pub material: usize,

    pub property_changed: bool,
    pub transform_changed: bool,
    pub uniform_changed: bool,
    pub datas: HashMap<String, MeshData>,
    pub vector_fields: HashMap<String, VectorFieldData>,

    //cuz we have to build a renderer from the main loop after we add them
    pub added_vector_fields: Vec<(String, Vec<[f32; 3]>, Vec<[f32; 3]>)>,

    pub shown_data: Option<String>,
    pub data_to_show: Option<Option<String>>,
    // When the data to show is changed, so we can do the change from the main loop
    pub color: [f32; 4],
    pub show: bool,
    pub show_edges: bool,
    pub smooth: bool,
    pub show_gizmo: bool,
    pub gizmo_mode: egui_gizmo::GizmoMode,
    pub transform: Transform,
}

//TODO some kind of error handling
//TODO refactor datas properties
//split part to copy when replacing the vertices
//copy data by part : if mapped to vertices, keep if nv same, idem with faces
/*

struct MeshState {

}

struct MeshUISettings {
    pub color: [f32; 4],
    pub show: bool,
    pub show_edges: bool,
    pub smooth: bool,
    pub show_gizmo: bool,
    pub gizmo_mode: egui_gizmo::GizmoMode,
    pub transform: Transform,
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
    pub fn new(name: &str, vertices: &Vec<[f32; 3]>, indices: &Vec<Vec<u32>>) -> Self {
        let normals = Self::compute_normals(vertices, indices);
        let face_normals = Self::compute_face_normals(vertices, indices);
        let vertices = vertices.clone();
        let internal_vertices = vertices
            .iter()
            .zip(normals)
            .map(|(vertex, normal)| ModelVertex {
                position: [vertex[0], vertex[1], vertex[2]],
                tex_coords: [0., 0.],
                normal,
                face_normal: normal,
                color: [0.2, 0.2, 0.8],
                barycentric_coords: [0., 0., 0.],
                distance: 0.,
            })
            .collect::<Vec<_>>();
        let mut internal_indices = Vec::new();
        for face in indices {
            for i in 1..face.len() - 1 {
                internal_indices.push([face[0], face[i], face[i + 1]]);
            }
        }
        let indices = indices.clone();
        let transform = Transform::default();
        let num_elements = indices.len() as u32;
        Self {
            name: name.to_string(),
            vertices,
            internal_vertices,
            face_normals,
            indices,
            internal_indices,
            num_elements,
            material: 0,
            transform,
            transform_changed: false,
            uniform_changed: false,
            datas: HashMap::new(),
            vector_fields: HashMap::new(),
            added_vector_fields: Vec::new(),
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

    fn compute_normals(vertices: &Vec<[f32; 3]>, indices: &Vec<Vec<u32>>) -> Vec<[f32; 3]> {
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

    fn compute_face_normals(vertices: &Vec<[f32; 3]>, indices: &Vec<Vec<u32>>) -> Vec<[f32; 3]> {
        let mut normals = vec![[0., 0., 0.]; indices.len()];
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
        let settings = if let Some(mesh_data) = self.datas.remove(&name) {
            if let MeshData::VertexScalar(_datas, settings) = mesh_data {
                settings
            } else {
                crate::model::data::VertexScalarSettings::default()
            }
        } else {
            crate::model::data::VertexScalarSettings::default()
        };
        self.datas
            .insert(name, MeshData::VertexScalar(datas, settings));
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
        let settings = if let Some(mesh_data) = self.datas.remove(&name) {
            if let MeshData::UVMap(_datas, settings) = mesh_data {
                settings
            } else if let MeshData::UVCornerMap(_datas, settings) = mesh_data {
                settings
            } else {
                crate::model::data::UVMapSettings::default()
            }
        } else {
            crate::model::data::UVMapSettings::default()
        };
        self.datas.insert(name, MeshData::UVMap(datas, settings));
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
        let settings = if let Some(mesh_data) = self.datas.remove(&name) {
            if let MeshData::UVMap(_datas, settings) = mesh_data {
                settings
            } else if let MeshData::UVCornerMap(_datas, settings) = mesh_data {
                settings
            } else {
                crate::model::data::UVMapSettings::default()
            }
        } else {
            crate::model::data::UVMapSettings::default()
        };

        self.datas
            .insert(name, MeshData::UVCornerMap(datas, settings));
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

    //TODO save settings when already existing
    pub fn add_vertex_vector_field(&mut self, name: String, vectors: Vec<[f32; 3]>) -> &mut Self {
        assert!(vectors.len() == self.vertices.len());
        let vectors_offsets: Vec<[f32; 3]> = self.vertices.clone();
        self.added_vector_fields
            .push((name, vectors, vectors_offsets));
        self
    }

    //TODO save settings when already existing
    pub fn add_face_vector_field(&mut self, name: String, vectors: Vec<[f32; 3]>) -> &mut Self {
        assert!(vectors.len() == self.indices.len());
        let vectors_offsets: Vec<[f32; 3]> = self
            .indices
            .iter()
            .map(|face| {
                let mut res0 = 0.;
                let mut res1 = 0.;
                let mut res2 = 0.;
                for index in face {
                    let vertex = self.vertices[*index as usize];
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
        self.added_vector_fields
            .push((name, vectors, vectors_offsets));
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

impl Deref for Model {
    type Target = Mesh;

    fn deref(&self) -> &<Self as Deref>::Target {
        &self.mesh
    }
}

impl DerefMut for Model {
    fn deref_mut(&mut self) -> &mut <Self as Deref>::Target {
        &mut self.mesh
    }
}

impl Model {
    pub fn new(
        name: &str,
        vertices: &Vec<[f32; 3]>,
        indices: &Vec<Vec<u32>>,
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
        for (name, vectors, vectors_offsets) in self.mesh.added_vector_fields.drain(..) {
            let vector_field_settings = VectorFieldSettings {
                magnitude: 1.,
                _padding: [0; 7],
            };
            let field = VectorField::new(
                device,
                camera_light_bind_group_layout,
                &self.renderer.transform_bind_group_layout,
                color_format,
                vectors,
                vectors_offsets,
                vector_field_settings,
            );
            let shown = if let Some(vector_field_data) = self.mesh.vector_fields.remove(&name) {
                vector_field_data.shown
            } else {
                true
            };
            let vector_field = VectorFieldData { field, shown };
            self.mesh.vector_fields.insert(name, vector_field);
        }
        for (_name, vector_field) in &mut self.mesh.vector_fields {
            vector_field.field.update(queue);
        }
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

    pub fn get_element_info(&self, element: usize) -> (String, usize) {
        if element < self.mesh.vertices.len() {
            ("vertex".into(), element)
        } else if element < self.mesh.vertices.len() + self.mesh.indices.len() {
            ("face".into(), element - self.mesh.vertices.len())
        } else {
            ("".into(), element)
        }
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
            for vector_field in model.mesh.vector_fields.values() {
                if vector_field.shown {
                    vector_field.field.render(self);
                }
            }
        }
    }

    fn draw_picker(&mut self, model: &'b Model) {
        if model.mesh.show {
            model.picker.render(self);
        }
    }
}
