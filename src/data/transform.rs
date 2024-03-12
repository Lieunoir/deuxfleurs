use crate::ui::UiDataElement;

use egui_gizmo::GizmoMode;

pub struct TransformSettings {
    pub transform: [[f32; 4]; 4],
    show_gizmo: bool,
    gizmo_mode: GizmoMode,
}

impl TransformSettings {
    pub fn to_raw(&self) -> TransformRaw {
        use cgmath::Matrix;
        use cgmath::SquareMatrix;
        let model = self.transform;
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
    pub fn draw_transform(&mut self, ui: &mut egui::Ui, positions: &[[f32; 3]]) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_gizmo, "Show Gizmo");
            egui::ComboBox::from_label("Guizmo Mode")
                .selected_text(format!("{:?}", self.gizmo_mode))
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.gizmo_mode, GizmoMode::Translate, "Translate")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.gizmo_mode, GizmoMode::Rotate, "Rotate")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.gizmo_mode, GizmoMode::Scale, "Scale")
                        .changed();
                });
        });
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Reset")).clicked() {
                self.transform = [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ];
                changed = true;
            }
            if ui.add(egui::Button::new("Center")).clicked() {
                let mut min_x = std::f32::MAX;
                let mut min_y = std::f32::MAX;
                let mut min_z = std::f32::MAX;
                let mut max_x = std::f32::MIN;
                let mut max_y = std::f32::MIN;
                let mut max_z = std::f32::MIN;

                let transfo_matrix: cgmath::Matrix4<f32> = self.transform.into();
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
                self.transform[3][0] -= x;
                self.transform[3][1] -= y;
                self.transform[3][2] -= z;
                changed = true;
            }
            if ui.add(egui::Button::new("Unit Scale")).clicked() {
                let mut min_x = std::f32::MAX;
                let mut min_y = std::f32::MAX;
                let mut min_z = std::f32::MAX;
                let mut max_x = std::f32::MIN;
                let mut max_y = std::f32::MIN;
                let mut max_z = std::f32::MIN;

                let transfo_matrix: cgmath::Matrix4<f32> = self.transform.into();
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
                for (i, row) in self.transform.iter_mut().enumerate() {
                    for value in row.iter_mut() {
                        if i < 3 {
                            *value *= scale;
                        }
                    }
                }
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
    //pub fn new() -> Self {
    //
    //}

    fn default() -> Self {
        TransformSettings {
            transform: [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            show_gizmo: false,
            gizmo_mode: GizmoMode::Translate,
        }
    }
}

impl UiDataElement for TransformSettings {
    fn draw(&mut self, ui: &mut egui::Ui, _property_changed: &mut bool) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_gizmo, "Show Gizmo");
            egui::ComboBox::from_label("Guizmo Mode")
                .selected_text(format!("{:?}", self.gizmo_mode))
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(&mut self.gizmo_mode, GizmoMode::Translate, "Translate")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.gizmo_mode, GizmoMode::Rotate, "Rotate")
                        .changed();
                    changed |= ui
                        .selectable_value(&mut self.gizmo_mode, GizmoMode::Scale, "Scale")
                        .changed();
                });
        });
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new("Reset")).clicked() {
                self.transform = [
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.],
                ];
                changed = true;
            }
        });
        changed
    }

    fn draw_gizmo(
        &mut self,
        ui: &mut egui::Ui,
        name: &str,
        view: cgmath::Matrix4<f32>,
        proj: cgmath::Matrix4<f32>,
    ) -> bool {
        if self.show_gizmo {
            let gizmo = egui_gizmo::Gizmo::new(format!("{} gizmo", name))
                .view_matrix(view)
                .projection_matrix(proj)
                .model_matrix(self.transform)
                .viewport(ui.clip_rect())
                .mode(self.gizmo_mode);
            let last_gizmo_response = gizmo.interact(ui);

            if let Some(gizmo_response) = last_gizmo_response {
                self.transform = gizmo_response.transform().to_cols_array_2d();
                true
            } else {
                false
            }
        } else {
            false
        }
    }
}
