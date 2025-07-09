use crate::shape::{Context, DataMut, DataMutTrait};
use crate::ui::UiDataElement;
use egui::Widget;
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Serialize, Deserialize)]
pub struct UVMapSettings {
    color_1: [f32; 4],
    color_2: [f32; 4],
    frequency: f32,
    #[serde(skip)]
    _padding: [f32; 3],
}

impl Default for UVMapSettings {
    fn default() -> Self {
        Self {
            color_1: [0.9, 0.9, 0.9, 1.],
            color_2: [0.6, 0.2, 0.4, 1.],
            frequency: 20.,
            _padding: [0.; 3],
        }
    }
}

impl UiDataElement for UVMapSettings {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        //ui.add(egui::Slider::new(&mut self.frequency, 0.0..=100.0).text("Period"));
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
            .on_hover_text("Checkerboard color 1")
            .changed();
            changed |= egui::widgets::color_picker::color_edit_button_rgba(
                ui,
                &mut color_2,
                egui::widgets::color_picker::Alpha::Opaque,
            )
            .on_hover_text("Checkerboard color 2")
            .changed();
            changed |= egui::DragValue::new(&mut self.frequency)
                .prefix("Frequency: ")
                .ui(ui)
                .changed();
        });
        self.color_1 = color_1.to_array();
        self.color_2 = color_2.to_array();
        changed
    }
}

pub type UVMapSettingsMut<'a, Ctxt> = DataMut<'a, &'a mut UVMapSettings, Ctxt>;

impl<'a, Ctxt: Context> UVMapSettingsMut<'a, Ctxt>
where
    Self: DataMutTrait,
{
    pub fn set_color_1(&mut self, color: [f32; 4]) {
        self.inner.color_1 = color;
        self.update_data_settings();
    }

    pub fn set_color_2(&mut self, color: [f32; 4]) {
        self.inner.color_2 = color;
        self.update_data_settings();
    }

    pub fn set_pattern_frequency(&mut self, frequency: f32) {
        self.inner.frequency = frequency;
        self.update_data_settings();
    }
}
