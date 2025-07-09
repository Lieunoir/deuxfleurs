use crate::ui::UiDataElement;
use egui::Widget;
#[cfg(feature = "saves")]
use serde::{Deserialize, Serialize};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "saves", derive(Serialize, Deserialize))]
pub struct IsolineSettings {
    pub isoline_number: f32,
    #[cfg_attr(feature = "saves", serde(skip))]
    _padding: [f32; 3],
}

impl Default for IsolineSettings {
    fn default() -> Self {
        Self {
            isoline_number: 0.,
            _padding: [0.; 3],
        }
    }
}

impl UiDataElement for IsolineSettings {
    fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            changed = egui::DragValue::new(&mut self.isoline_number)
                .prefix("Isolines: ")
                .ui(ui)
                .changed();
        });
        changed
    }
}
