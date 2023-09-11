use std::io::{BufReader, Cursor};

use cfg_if::cfg_if;

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let base = reqwest::Url::parse(&format!(
        "{}/{}/",
        location.origin().unwrap(),
        option_env!("RES_PATH").unwrap_or("assets"),
    ))
    .unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            log::warn!("Load model on web");

            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;

            log::warn!("{}", txt);

        } else {
            let path = std::path::Path::new("assets")
                .join(file_name);
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_mesh(file_name: &str) -> anyhow::Result<(Vec<[f32; 3]>, Vec<Vec<u32>>)> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, _obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: false,
            single_index: false,
            ..Default::default()
        },
        |p| async move {
            if let Ok(mat_text) = load_string(&p).await {
                tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
            } else {
                Ok((Vec::new(), ahash::AHashMap::new()))
            }
        },
    )
    .await?;
    let mesh = &models.get(0).unwrap().mesh;
    let vertices = mesh
        .positions
        .chunks(3)
        .map(|vertex| vertex.try_into().unwrap())
        .collect::<Vec<_>>();
    let mut indices = Vec::new();
    let mut i = 0;
    if mesh.face_arities.len() > 0 {
        for face_arity in &mesh.face_arities {
            indices.push(mesh.indices[i..i + *face_arity as usize].into());
            i += *face_arity as usize;
        }
    } else {
        indices = mesh
            .indices
            .chunks(3)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<_>>();
    }
    Ok((vertices, indices))
}

pub async fn load_preloaded_mesh(data: Vec<u8>) -> anyhow::Result<(Vec<[f32; 3]>, Vec<Vec<u32>>)> {
    let obj_cursor = Cursor::new(data);
    let mut obj_reader = BufReader::new(obj_cursor);

    let (models, _obj_materials) = tobj::load_obj_buf_async(
        &mut obj_reader,
        &tobj::LoadOptions {
            triangulate: false,
            single_index: false,
            ..Default::default()
        },
        |p| async move {
            if let Ok(mat_text) = load_string(&p).await {
                tobj::load_mtl_buf(&mut BufReader::new(Cursor::new(mat_text)))
            } else {
                Ok((Vec::new(), ahash::AHashMap::new()))
            }
        },
    )
    .await?;
    let mesh = &models.get(0).unwrap().mesh;
    let vertices = mesh
        .positions
        .chunks(3)
        .map(|vertex| vertex.try_into().unwrap())
        .collect::<Vec<_>>();
    let mut indices = Vec::new();
    let mut i = 0;
    if mesh.face_arities.len() > 0 {
        for face_arity in &mesh.face_arities {
            indices.push(mesh.indices[i..i + *face_arity as usize].into());
            i += *face_arity as usize;
        }
    } else {
        indices = mesh
            .indices
            .chunks(3)
            .map(|face| face.try_into().unwrap())
            .collect::<Vec<_>>();
    }
    Ok((vertices, indices))
}
