use serde::{Deserialize, Serialize};

// Actually more of a cylinder, we don't really use the y
// coordinate
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct SBV {
    pub center: [f32; 3],
    pub radius: f32,
}

impl SBV {
    pub fn new(points: &[[f32; 3]]) -> Self {
        let x = points[0];
        let mut max_dist = 0.;
        let mut y = [0., 0., 0.];
        for p in points {
            let dist = ((x[0] - p[0]).powi(2) + (x[2] - p[2]).powi(2)).sqrt();
            if dist > max_dist {
                y = *p;
                max_dist = dist;
            }
        }
        max_dist = 0.;
        let mut z = [0., 0., 0.];
        for p in points {
            let dist = ((y[0] - p[0]).powi(2) + (y[2] - p[2]).powi(2)).sqrt();
            if dist > max_dist {
                z = *p;
                max_dist = dist;
            }
        }

        let center = [
            (y[0] + z[0]) * 0.5,
            (y[1] + z[1]) * 0.5,
            (y[2] + z[2]) * 0.5,
        ];
        let radius = max_dist * 0.5;
        let mut res = SBV { center, radius };
        for point in points {
            res.add_point(*point);
        }

        res
    }

    pub fn add_point(&mut self, point: [f32; 3]) {
        let d_x = self.center[0] - point[0];
        let d_z = self.center[2] - point[2];
        let dist = (d_x * d_x + d_z * d_z).sqrt();
        if dist > self.radius {
            self.center[0] =
                (self.center[0] + point[0] + (self.center[0] - point[0]) * self.radius / dist) / 2.;
            self.center[2] =
                (self.center[2] + point[2] + (self.center[2] - point[2]) * self.radius / dist) / 2.;
            self.radius = 0.5 * (self.radius + dist);
        }
    }

    pub fn merge(box1: &mut Option<SBV>, box2: &SBV) {
        if let Some(box1) = box1 {
            let d_x = box1.center[0] - box2.center[0];
            let d_z = box1.center[2] - box2.center[2];
            let dist = (d_x * d_x + d_z * d_z).sqrt();
            if dist + box1.radius <= box2.radius {
                box1.center = box2.center;
                box1.radius = box2.radius;
            } else if dist + box2.radius <= box1.radius {
                ()
            } else {
                box1.center[0] = (box1.center[0]
                    + box2.center[0]
                    + (box1.center[0] - box2.center[0]) * (box1.radius - box2.radius) / dist)
                    / 2.;
                box1.center[2] = (box1.center[2]
                    + box2.center[2]
                    + (box1.center[2] - box2.center[2]) * (box1.radius - box2.radius) / dist)
                    / 2.;
                box1.radius = 0.5 * (box1.radius + box2.radius + dist);
            }
        } else {
            *box1 = Some(box2.clone());
        }
    }

    pub fn get_bb(&self) -> [f32; 4] {
        [
            self.center[0] - self.radius,
            self.center[0] + self.radius,
            self.center[2] - self.radius,
            self.center[2] + self.radius,
        ]
    }

    pub fn transform(&self, transform: &[[f32; 4]; 4]) -> Self {
        let transform = glam::Mat4::from_cols_array_2d(transform);
        let center = transform.project_point3(self.center.into()).into();

        let volume = transform.x_axis.truncate().length().max(
            transform
                .y_axis
                .truncate()
                .length()
                .max(transform.z_axis.truncate().length()),
        );
        let radius = self.radius * volume;
        Self { center, radius }
    }
}
