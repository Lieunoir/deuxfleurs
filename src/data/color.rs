use crate::ui::UiDataElement;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ColorSettings {
    pub color: [f32; 4],
}

impl Default for ColorSettings {
    fn default() -> Self {
        Self {
            color: [0.2, 0.2, 0.8, 1.],
        }
    }
}

impl UiDataElement for ColorSettings {
    fn draw(&mut self, ui: &mut egui::Ui) -> bool {
        let mut mesh_color = egui::Rgba::from_rgba_unmultiplied(
            self.color[0],
            self.color[1],
            self.color[2],
            self.color[3],
        );
        let mut changed = false;
        ui.horizontal(|ui| {
            changed = egui::widgets::color_picker::color_edit_button_rgba(
                ui,
                &mut mesh_color,
                egui::widgets::color_picker::Alpha::Opaque,
            )
            .changed();
            ui.label("Color");
        });
        self.color = mesh_color.to_array();
        changed
    }
}
