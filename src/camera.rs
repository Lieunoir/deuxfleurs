use crate::{Settings, aabb::SBV};

use serde::{Deserialize, Serialize};
use winit::event::*;

// Camera informations for easier updating
#[derive(Clone, Serialize, Deserialize)]
pub struct Camera {
    eye: glam::Vec3,
    target: glam::Vec3,
    up: glam::Vec3,
    #[serde(skip)]
    aspect: f32,
    #[serde(skip)]
    fovy: f32,
    #[serde(skip)]
    znear: f32,
    #[serde(skip)]
    zfar: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: (0.0, 0.0, -2.5).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: glam::Vec3::new(0., 1., 0.),
            aspect,
            fovy: 45.0,
            znear: 0.01,
            zfar: 10.0,
        }
    }

    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
        let proj = glam::Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar);
        proj * view
    }

    pub fn build_view(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    pub fn build_proj(&self) -> glam::Mat4 {
        glam::Mat4::perspective_rh(self.fovy, self.aspect, self.znear, self.zfar)
    }

    pub fn set_scene_size(&mut self, size: f32, center: glam::Vec3) {
        if size > 0. {
            let dir = glam::Vec3::new(0., 0., -3.);
            self.eye = center + dir * size;
            self.target = center;
            self.znear = size * 0.01;
            self.zfar = size * 10.;
            self.up = glam::Vec3::new(0., 1., 0.);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn get_position(&self) -> [f32; 3] {
        self.eye.into()
    }

    pub fn copy(&self) -> Result<String, ()> {
        serde_json::to_string(self).map_err(|_e| ())
    }

    pub fn set_from_camera(&mut self, new_camera: Camera) {
        self.eye = new_camera.eye;
        self.target = new_camera.target;
        self.up = new_camera.up;
    }

    pub fn set(&mut self, new_camera: String) {
        if let Ok(new_c) = serde_json::from_str::<Camera>(&new_camera) {
            self.set_from_camera(new_c);
        }
    }
}

// Camera matrices for GPU computations
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
    view_inv: [[f32; 4]; 4],
    floor_bb: [f32; 4],
    floor_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            view_inv: glam::Mat4::IDENTITY.to_cols_array_2d(),
            floor_bb: [0.0; 4],
            floor_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, sbv: &SBV, level: f32) {
        // We're using Vector4 because ofthe camera_uniform 16 byte spacing requirement
        self.view_position = camera.eye.extend(1.).into();
        let view_proj = camera.build_view_projection_matrix();
        self.view_proj = view_proj.to_cols_array_2d();
        let view_inv = view_proj.inverse();
        self.view_inv = view_inv.to_cols_array_2d();
        //let orig : cgmath::Vector4<f32> = self.view_position.into();
        let mut min_x = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_z = f32::MIN;
        let couples = [(-1., -1.), (-1., 1.), (1., -1.), (1., 1.)];
        for (x, y) in couples {
            let mut target = view_inv * glam::Vec4::new(x, y, 1., 1.);
            target = target / target.w;
            let mut origin = view_inv * glam::Vec4::new(x, y, 0., 1.);
            origin = origin / origin.w;
            let ray = target - origin;
            if ray.y.abs() > 10e-9 {
                let t = (level - origin.y) / ray.y;
                if t <= 1. && t >= 0. {
                    let pos = origin + t * ray;
                    if pos.x < min_x {
                        min_x = pos.x;
                    }
                    if pos.x > max_x {
                        max_x = pos.x;
                    }
                    if pos.z < min_z {
                        min_z = pos.z;
                    }
                    if pos.z > max_z {
                        max_z = pos.z;
                    }
                }
            }
        }
        let couples = [(-1., 0.), (-1., 1.), (1., 0.), (1., 1.)];
        for (x, z) in couples {
            let mut target = view_inv * glam::Vec4::new(x, -1., z, 1.);
            target = target / target.w;
            let mut origin = view_inv * glam::Vec4::new(x, 1., z, 1.);
            origin = origin / origin.w;
            let ray = target - origin;
            if ray.y.abs() > 10e-9 {
                let t = (level - origin.y) / ray.y;
                if t <= 1. && t >= 0. {
                    let pos = origin + t * ray;
                    if pos.x < min_x {
                        min_x = pos.x;
                    }
                    if pos.x > max_x {
                        max_x = pos.x;
                    }
                    if pos.z < min_z {
                        min_z = pos.z;
                    }
                    if pos.z > max_z {
                        max_z = pos.z;
                    }
                }
            }
        }
        let couples = [(-1., 0.), (-1., 1.), (1., 0.), (1., 1.)];
        for (y, z) in couples {
            let mut target = view_inv * glam::Vec4::new(-1., y, z, 1.);
            target = target / target.w;
            let mut origin = view_inv * glam::Vec4::new(1., y, z, 1.);
            origin = origin / origin.w;
            let ray = target - origin;
            if ray.y.abs() > 10e-9 {
                let t = (level - origin.y) / ray.y;
                if t <= 1. && t >= 0. {
                    let pos = origin + t * ray;
                    if pos.x < min_x {
                        min_x = pos.x;
                    }
                    if pos.x > max_x {
                        max_x = pos.x;
                    }
                    if pos.z < min_z {
                        min_z = pos.z;
                    }
                    if pos.z > max_z {
                        max_z = pos.z;
                    }
                }
            }
        }
        let sbv_bb = sbv.get_bb();
        min_x = min_x.max(sbv_bb[0]);
        max_x = max_x.min(sbv_bb[1]);
        min_z = min_z.max(sbv_bb[2]);
        max_z = max_z.min(sbv_bb[3]);
        let c_x = 0.5 * (min_x + max_x);
        let c_z = 0.5 * (min_z + max_z);
        let d_x = 0.5 * (max_x - min_x);
        let d_z = 0.5 * (max_z - min_z);
        let eye = glam::Vec3::new(c_x, camera.zfar, c_z);
        let target = glam::Vec3::new(c_x, 0., c_z);
        let up = glam::Vec3::new(0., 0., 1.);
        let view = glam::Mat4::look_at_rh(eye, target, up);
        let proj =
            glam::Mat4::orthographic_rh(d_x, -d_x, -d_z, d_z, -camera.zfar, 2. * camera.zfar);
        self.floor_bb = [min_x, min_z, max_x - min_x, max_z - min_z];
        self.floor_proj = (proj * view).to_cols_array_2d();
    }
}

// TODO : abstract this into a trait implemented by different controllers (arcball, fps...)
pub struct CameraController {
    is_mouse_left_pressed: bool,
    is_mouse_right_pressed: bool,
    prev_mouse: Option<winit::dpi::PhysicalPosition<f64>>,
    wheel_delta: Option<f32>,
    pan_delta: Option<(f32, f32)>,
}

impl CameraController {
    pub fn new() -> Self {
        Self {
            is_mouse_left_pressed: false,
            is_mouse_right_pressed: false,
            prev_mouse: None,
            wheel_delta: None,
            pan_delta: None,
        }
    }

    pub fn process_events(&mut self, event: &WindowEvent, ui_hovered: bool) -> bool {
        match event {
            WindowEvent::Touch(touch_event) => match touch_event.phase {
                TouchPhase::Started => {
                    if self.is_mouse_left_pressed {
                        self.is_mouse_right_pressed = true;
                    } else {
                        self.is_mouse_left_pressed = true;
                    }
                    true
                }
                TouchPhase::Ended | TouchPhase::Cancelled => {
                    if self.is_mouse_right_pressed {
                        self.is_mouse_right_pressed = false;
                        self.prev_mouse = None;
                    } else {
                        self.is_mouse_left_pressed = false;
                        self.prev_mouse = None;
                    }
                    true
                }
                TouchPhase::Moved => {
                    if self.is_mouse_right_pressed {
                        if let Some(prev) = self.prev_mouse {
                            self.wheel_delta = Some(0.1 * (touch_event.location.y - prev.y) as f32);
                        }
                    } else {
                        if let Some(prev) = self.prev_mouse {
                            self.pan_delta = Some((
                                (touch_event.location.x - prev.x) as f32,
                                (touch_event.location.y - prev.y) as f32,
                            ));
                        }
                    }
                    self.prev_mouse = Some(touch_event.location);
                    true
                }
            },
            WindowEvent::CursorMoved { position, .. } if self.prev_mouse.is_none() => {
                self.prev_mouse = Some(*position);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                let prev = self.prev_mouse.unwrap();
                self.pan_delta = Some(((position.x - prev.x) as f32, (position.y - prev.y) as f32));
                self.prev_mouse = Some(*position);
                if self.is_mouse_left_pressed || self.is_mouse_right_pressed {
                    true
                } else {
                    false
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if !ui_hovered || *state != ElementState::Pressed {
                    if *button == MouseButton::Left {
                        self.is_mouse_left_pressed = *state == ElementState::Pressed;
                    } else if *button == MouseButton::Right {
                        self.is_mouse_right_pressed = *state == ElementState::Pressed;
                    }
                }
                false
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !ui_hovered {
                    self.wheel_delta = Some(match delta {
                        MouseScrollDelta::LineDelta(_, y) => *y,
                        MouseScrollDelta::PixelDelta(p) => 0.02 * p.y as f32,
                    });
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, settings: &Settings) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();

        if let Some(delta) = self.wheel_delta {
            camera.eye += forward_norm * delta * camera.zfar * 0.2 * settings.zoom_sensitivity;
            camera.target += forward_norm * delta * camera.zfar * 0.2 * settings.zoom_sensitivity;
            self.wheel_delta = None;
        }

        let right = forward_norm.cross(camera.up);
        if let Some((dx, dy)) = self.pan_delta {
            if self.is_mouse_left_pressed {
                let origin = glam::Vec3::new(0., 0., 0.);
                let eye_norm = (camera.eye - glam::Vec3::new(0., 0., 0.)).normalize();
                let eye_mag = (camera.eye - glam::Vec3::new(0., 0., 0.)).length();
                let center_right = eye_norm.cross(camera.up);
                let old_eye = camera.eye;
                camera.eye = glam::Vec3::new(0., 0., 0.)
                    + (eye_norm
                        + center_right * dx * 0.2 * settings.mouse_sensitivity
                        + camera.up * dy * 0.2 * settings.mouse_sensitivity)
                        .normalize()
                        * eye_mag;
                let rotation = glam::Quat::from_rotation_arc(
                    (old_eye - origin).normalize(),
                    (camera.eye - origin).normalize(),
                );
                camera.target = origin + rotation.mul_vec3(camera.target - origin);
                camera.up = rotation.mul_vec3(old_eye + camera.up - origin) - (camera.eye - origin);
                //camera.target = camera.eye + forward;
            } else if self.is_mouse_right_pressed {
                camera.eye += camera.zfar
                    * 0.006
                    * settings.mouse_sensitivity
                    * (-dx * right + dy * camera.up);
                camera.target += camera.zfar
                    * 0.006
                    * settings.mouse_sensitivity
                    * (-dx * right + dy * camera.up);
            }
            self.pan_delta = None;
        }

        // Redo radius calc in case the up/ down is pressed.
        /*
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and eye so
            // that it doesn't change. The eye therefore still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
        */
    }
}
