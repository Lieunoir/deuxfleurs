use crate::ui::UiDataElement;
use glam::Vec4Swizzles;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};
use transform_gizmo_egui::math::{DMat4, DQuat, DVec3, Transform};
use transform_gizmo_egui::prelude::*;

#[derive(Clone)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct TransformSettings {
    //pub transform: [[f64; 4]; 4],
    show_gizmo: bool,
    //gizmo_mode: GizmoMode,
    scale: DVec3,
    rotation: DQuat,
    translation: DVec3,
    #[cfg_attr(feature = "saves", serde(skip))]
    gizmo: Gizmo,
}

impl TransformSettings {
    pub fn get_transform(&self) -> [[f32; 4]; 4] {
        let model_m =
            DMat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation);
        let mut model: [[f32; 4]; 4] = [[0.; 4]; 4];
        for (row_m, row) in model_m.to_cols_array_2d().into_iter().zip(model.iter_mut()) {
            for (v_m, v) in row_m.into_iter().zip(row.iter_mut()) {
                *v = v_m as f32;
            }
        }
        model
    }

    pub fn to_raw(&self) -> TransformRaw {
        let mut normal: [[f32; 3]; 3] = [[0.; 3]; 3];
        let model = self.get_transform();
        for (row, row_orig) in normal.iter_mut().zip(model) {
            *row = row_orig[0..3].try_into().unwrap();
        }
        let mut normal = glam::Mat3::from_cols_array_2d(&normal);
        normal = normal.inverse().transpose();
        let normal = glam::Mat4::from_mat3(normal);
        TransformRaw {
            model,
            normal: normal.to_cols_array_2d(),
        }
    }

    pub fn draw_transform(&mut self, ui: &mut egui::Ui, positions: &[[f32; 3]]) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_gizmo, "Transform Gizmo");
        });
        ui.horizontal(|ui| {
            if ui.button("Center").clicked() {
                let mut min_x = std::f32::MAX;
                let mut min_y = std::f32::MAX;
                let mut min_z = std::f32::MAX;
                let mut max_x = std::f32::MIN;
                let mut max_y = std::f32::MIN;
                let mut max_z = std::f32::MIN;

                let model = self.get_transform();

                let transfo_matrix = glam::Mat4::from_cols_array_2d(&model);
                for position in positions {
                    let position = transfo_matrix.project_point3((*position).into());
                    if position[0] < min_x {
                        min_x = position[0];
                    }
                    if position[1] < min_y {
                        min_y = position[1];
                    }
                    if position[2] < min_z {
                        min_z = position[2];
                    }
                    if position[0] > max_x {
                        max_x = position[0];
                    }
                    if position[1] > max_y {
                        max_y = position[1];
                    }
                    if position[2] > max_z {
                        max_z = position[2];
                    }
                }
                let x = (max_x + min_x) / 2.;
                let y = (max_y + min_y) / 2.;
                //let y = min_y;
                let z = (max_z + min_z) / 2.;
                self.translation += DVec3::from_array([-x as f64, -y as f64, -z as f64]);
                changed = true;
            }
            if ui.button("Unit Scale").clicked() {
                let mut min_x = std::f32::MAX;
                let mut min_y = std::f32::MAX;
                let mut min_z = std::f32::MAX;
                let mut max_x = std::f32::MIN;
                let mut max_y = std::f32::MIN;
                let mut max_z = std::f32::MIN;

                let model = self.get_transform();

                let transfo_matrix = glam::Mat4::from_cols_array_2d(&model);
                for vertex in positions {
                    let position = transfo_matrix.project_point3((*vertex).into());

                    if position[0] < min_x {
                        min_x = position[0];
                    }
                    if position[1] < min_y {
                        min_y = position[1];
                    }
                    if position[2] < min_z {
                        min_z = position[2];
                    }
                    if position[0] > max_x {
                        max_x = position[0];
                    }
                    if position[1] > max_y {
                        max_y = position[1];
                    }
                    if position[2] > max_z {
                        max_z = position[2];
                    }
                }
                let x = max_x - min_x;
                let y = max_y - min_y;
                let z = max_z - min_z;
                let scale = 1. / (x * x + y * y + z * z).sqrt();
                let box_center_x = (max_x + min_x) / 2.;
                let box_center_y = (max_y + min_y) / 2.;
                let box_center_z = (max_z + min_z) / 2.;
                self.scale *= scale as f64;
                let model_center = transfo_matrix * glam::Vec4::new(0., 0., 0., 1.);
                let model_center = model_center.xyz() / model_center.w;
                self.translation += glam::DVec3::from_array([
                    -((1. - scale) * (box_center_x + model_center.x)) as f64,
                    -((1. - scale) * (box_center_y + model_center.y)) as f64,
                    -((1. - scale) * (box_center_z + model_center.z)) as f64,
                ]);
                changed = true;
            }
            if ui.button("Reset").clicked() {
                self.translation = DVec3::ZERO;
                self.scale = DVec3::ONE;
                self.rotation = DQuat::IDENTITY;
                changed = true;
            }
        });
        changed
    }

    pub(crate) fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        view: glam::Mat4,
        proj: glam::Mat4,
        gizmo_hovered: &mut bool,
    ) -> bool {
        if self.show_gizmo {
            let viewport = ui.clip_rect();
            let view_m = view.as_dmat4();
            let proj_m = proj.as_dmat4();
            self.gizmo.update_config(GizmoConfig {
                view_matrix: view_m.into(),
                projection_matrix: proj_m.into(),
                modes: GizmoMode::all()
                    .difference(GizmoMode::all_scale())
                    .difference(enum_set!(GizmoMode::TranslateView))
                    .difference(enum_set!(GizmoMode::RotateView))
                    .union(enum_set!(GizmoMode::ScaleUniform)),
                orientation: GizmoOrientation::Local,
                viewport,
                ..Default::default()
            });

            let mut transform = Transform::from_scale_rotation_translation(
                self.scale,
                self.rotation,
                self.translation,
            );

            let res = if let Some((_result, new_transforms)) = self.gizmo.interact(ui, &[transform])
            {
                for (new_transform, transform) in
                    new_transforms.iter().zip(std::iter::once(&mut transform))
                {
                    // Apply the modified transforms
                    *transform = *new_transform;
                }
                self.scale = transform.scale.into();
                self.rotation = transform.rotation.into();
                self.translation = transform.translation.into();

                true
            } else {
                false
            };
            *gizmo_hovered |= self.gizmo.is_focused();
            res
        } else {
            false
        }
    }

    pub(crate) fn get_local_transform(&self, vertex: [f32; 3]) -> Transform {
        let pos = glam::Vec3::from_array(vertex);
        let pos = pos.as_dvec3();
        let pos = self.scale * pos;
        let pos = self.rotation * pos;
        Transform::from_scale_rotation_translation(
            self.scale,
            self.rotation,
            self.translation + pos,
        )
    }

    pub(crate) fn reverse_local_transform(&self, translation: DVec3) -> [f32; 3] {
        let pos = translation - self.translation;
        let pos = self.rotation.inverse() * pos;
        let pos = pos / self.scale;
        pos.as_vec3().into()
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

impl Default for TransformSettings {
    fn default() -> Self {
        TransformSettings {
            translation: DVec3::ZERO,
            scale: DVec3::ONE,
            rotation: DQuat::IDENTITY,
            show_gizmo: false,
            gizmo: Gizmo::default(),
        }
    }
}

impl UiDataElement for TransformSettings {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_gizmo, "Transform Gizmo");
        });
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Reset Transform")).clicked() {
                self.translation = DVec3::ZERO;
                self.scale = DVec3::ONE;
                self.rotation = DQuat::IDENTITY;
                changed = true;
            }
        });
        changed
    }
}
