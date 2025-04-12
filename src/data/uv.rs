use crate::ui::UiDataElement;
use egui::Widget;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UVMapSettings {
    pub color_1: [f32; 4],
    pub color_2: [f32; 4],
    pub period: f32,
    _padding: [f32; 3],
}

impl Default for UVMapSettings {
    fn default() -> Self {
        Self {
            color_1: [0.9, 0.9, 0.9, 1.],
            color_2: [0.6, 0.2, 0.4, 1.],
            period: 20.,
            _padding: [0.; 3],
        }
    }
}

impl UiDataElement for UVMapSettings {
    fn draw(&mut self, ui: &mut egui::Ui, _property_changed: &mut bool) -> bool {
        let mut changed = false;
        //ui.add(egui::Slider::new(&mut self.period, 0.0..=100.0).text("Period"));
        changed |= egui::Slider::new(&mut self.period, 0.0..=100.0)
            .text("Period")
            .clamping(egui::SliderClamping::Never)
            .ui(ui)
            .changed();
        let mut color_1 = egui::Rgba::from_rgba_unmultiplied(
            self.color_1[0],
            self.color_1[1],
            self.color_1[2],
            self.color_1[3],
        );
        let mut color_2 = egui::Rgba::from_rgba_unmultiplied(
            self.color_2[0],
            self.color_2[1],
            self.color_2[2],
            self.color_2[3],
        );
        ui.horizontal(|ui| {
            changed |= egui::widgets::color_picker::color_edit_button_rgba(
                ui,
                &mut color_1,
                egui::widgets::color_picker::Alpha::Opaque,
            )
            .changed();
            ui.label("Checkerboard color 1");
        });
        ui.horizontal(|ui| {
            changed |= egui::widgets::color_picker::color_edit_button_rgba(
                ui,
                &mut color_2,
                egui::widgets::color_picker::Alpha::Opaque,
            )
            .changed();
            ui.label("Checkerboard color 2");
        });
        self.color_1 = color_1.to_array();
        self.color_2 = color_2.to_array();
        changed
    }
}
