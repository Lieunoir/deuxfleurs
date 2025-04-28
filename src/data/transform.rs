use crate::ui::UiDataElement;
use transform_gizmo_egui::math::{DMat4, DQuat, DVec3, Transform};
use transform_gizmo_egui::prelude::*;

pub struct TransformSettings {
    //pub transform: [[f64; 4]; 4],
    show_gizmo: bool,
    //gizmo_mode: GizmoMode,
    scale: DVec3,
    rotation: DQuat,
    translation: DVec3,
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
        use cgmath::Matrix;
        use cgmath::SquareMatrix;
        //let model = self.transform;
        let mut normal: [[f32; 3]; 3] = [[0.; 3]; 3];
        let model = self.get_transform();
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
    pub fn draw_transform(&mut self, ui: &mut egui::Ui, positions: &[[f32; 3]]) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_gizmo, "Show Gizmo");
        });
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Reset")).clicked() {
                self.translation = DVec3::ZERO;
                self.scale = DVec3::ONE;
                self.rotation = DQuat::IDENTITY;
                changed = true;
            }
            if ui.add(egui::Button::new("Center")).clicked() {
                let mut min_x = std::f32::MAX;
                let mut min_y = std::f32::MAX;
                let mut min_z = std::f32::MAX;
                let mut max_x = std::f32::MIN;
                let mut max_y = std::f32::MIN;
                let mut max_z = std::f32::MIN;

                let model = self.get_transform();

                let transfo_matrix: cgmath::Matrix4<f32> = model.into();
                for position in positions {
                    let threed_point: cgmath::Point3<f32> = (*position).into();

                    let position = cgmath::Point3::<f32>::from_homogeneous(
                        transfo_matrix * threed_point.to_homogeneous(),
                    );
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
                //let y = (max_y + min_y) / 2.;
                let y = min_y;
                let z = (max_z + min_z) / 2.;
                self.translation += DVec3::from_array([-x as f64, -y as f64, -z as f64]);
                changed = true;
            }
            if ui.add(egui::Button::new("Unit Scale")).clicked() {
                let mut min_x = std::f32::MAX;
                let mut min_y = std::f32::MAX;
                let mut min_z = std::f32::MAX;
                let mut max_x = std::f32::MIN;
                let mut max_y = std::f32::MIN;
                let mut max_z = std::f32::MIN;

                let model = self.get_transform();

                let transfo_matrix: cgmath::Matrix4<f32> = model.into();
                for vertex in positions {
                    let threed_point: cgmath::Point3<f32> = (*vertex).into();

                    let position = cgmath::Point3::<f32>::from_homogeneous(
                        transfo_matrix * threed_point.to_homogeneous(),
                    );
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
                self.scale *= scale as f64;
                changed = true;
            }
        });
        changed
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
    fn draw(&mut self, ui: &mut egui::Ui, _property_changed: &mut bool) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_gizmo, "Show Gizmo");
        });
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Reset")).clicked() {
                self.translation = DVec3::ZERO;
                self.scale = DVec3::ONE;
                self.rotation = DQuat::IDENTITY;
                changed = true;
            }
        });
        changed
    }

    fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
        gizmo_hovered: &mut bool,
    ) -> bool {
        if self.show_gizmo {
            let viewport = ui.clip_rect();
            let view: [[f32; 4]; 4] = view.into();
            let proj: [[f32; 4]; 4] = proj.into();
            let mut view_m = DMat4::ZERO;
            let mut proj_m = DMat4::ZERO;
            for (i, (row_m, row)) in view_m
                .to_cols_array_2d()
                .iter_mut()
                .zip(view.into_iter())
                .enumerate()
            {
                for (v_m, v) in row_m.iter_mut().zip(row.into_iter()) {
                    *v_m = v as f64;
                }
                match i {
                    0 => view_m.x_axis = (*row_m).into(),
                    1 => view_m.y_axis = (*row_m).into(),
                    2 => view_m.z_axis = (*row_m).into(),
                    _ => view_m.w_axis = (*row_m).into(),
                }
            }
            for (i, (row_m, row)) in proj_m
                .to_cols_array_2d()
                .iter_mut()
                .zip(proj.into_iter())
                .enumerate()
            {
                for (v_m, v) in row_m.iter_mut().zip(row.into_iter()) {
                    *v_m = v as f64;
                }
                match i {
                    0 => proj_m.x_axis = (*row_m).into(),
                    1 => proj_m.y_axis = (*row_m).into(),
                    2 => proj_m.z_axis = (*row_m).into(),
                    _ => proj_m.w_axis = (*row_m).into(),
                }
            }
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
}
