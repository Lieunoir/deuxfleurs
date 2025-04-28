use deuxfleurs::{load_mesh, RunningState, Settings, StateHandle};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let mut handle = deuxfleurs::init();
    let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    handle.register_surface("spot".into(), spot_v, spot_f);

    let mut last_selected = 0;
    let mut last_selected_geometry = "".into();
    let callback = move |ui: &mut egui::Ui, state: &mut RunningState| {
        ui.label("Click on spot!");
        if let Some((surface_name, item)) = state.get_picked().clone() {
            if last_selected != item || last_selected_geometry != *surface_name {
                if let Some(surface) = state.get_surface_mut(&surface_name) {
                    let n_v = surface.geometry().vertices.len();
                    if item < n_v {
                        let mut selected = vec![0.; n_v];
                        last_selected = item;
                        last_selected_geometry = surface_name.clone();
                        selected[item] = 1.;
                        surface.add_vertex_scalar("selected vertex".into(), selected);
                    } else {
                        let mut selected = vec![0.; surface.geometry().indices.size()];
                        last_selected = item;
                        last_selected_geometry = surface_name.clone();
                        selected[item - n_v] = 1.;
                        surface.add_face_scalar("selected face".into(), selected);
                    }
                }
            }
        }
    };
    handle.run(
        1080,
        720,
        Some("deuxfleurs".into()),
        Settings::default(),
        callback,
    );
}
