use deuxfleurs::{
    RunningState, Settings, load_mesh,
    picker::{Picked, SurfacePicked},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

fn main() {
    pollster::block_on(run());
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    let mut handle = deuxfleurs::init(Settings::default());
    let (spot_v, spot_f) = load_mesh("examples/assets/spot.obj").await.unwrap();
    let num_edges = spot_v.len() + spot_f.size() - 2;
    handle.register_surface("spot", spot_v, spot_f);

    let mut last_selected = None;
    let mut last_selected_geometry = "".into();
    let callback = move |ui: &mut egui::Ui, state: &mut RunningState| {
        ui.label("Click on spot!");
        if let Some((surface_name, item)) = state.get_picked().clone() {
            if last_selected.as_ref() != Some(&item) || last_selected_geometry != *surface_name {
                if let Some(mut surface) = state.get_surface_mut(&surface_name) {
                    last_selected = Some(item.clone());
                    last_selected_geometry = surface_name.clone();
                    let n_v = surface.geometry().vertices.len();
                    match item {
                        Picked::Surface(SurfacePicked::Vertex(item)) => {
                            let mut selected = vec![0.; n_v];
                            selected[item as usize] = 1.;
                            surface.add_vertex_scalar("selected vertex", selected);
                        }
                        Picked::Surface(SurfacePicked::Face(item)) => {
                            let mut selected = vec![0.; surface.geometry().indices.size()];
                            selected[item as usize] = 1.;
                            surface.add_face_scalar("selected face", selected);
                        }
                        Picked::Surface(SurfacePicked::Edge(item)) => {
                            let mut selected = vec![0.; num_edges];
                            selected[item as usize] = 1.;
                            surface.add_edge_scalar("selected edge", selected);
                        }
                        _ => {}
                    }
                }
            }
        }
    };
    handle.run(1080, 720, Some("deuxfleurs"), callback);
}
