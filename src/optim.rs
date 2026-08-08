use rand::Rng;
use rand_distr::Normal;

use crate::types::*;

/// Compute mean x/y, sample standard deviation x/y, and Pearson correlation.
pub fn compute_stats(data: &Data) -> (f32, f32, f32, f32, f32) {
    assert_eq!(data.x.len(), data.y.len(), "x/y length mismatch");
    assert!(data.x.len() >= 2, "at least two points are required");
    let n = data.x.len() as f64;
    let mean_x = data.x.iter().map(|value| *value as f64).sum::<f64>() / n;
    let mean_y = data.y.iter().map(|value| *value as f64).sum::<f64>() / n;
    let mut variance_x = 0.0_f64;
    let mut variance_y = 0.0_f64;
    let mut covariance = 0.0_f64;
    for (&x, &y) in data.x.iter().zip(data.y.iter()) {
        let centered_x = x as f64 - mean_x;
        let centered_y = y as f64 - mean_y;
        variance_x += centered_x * centered_x;
        variance_y += centered_y * centered_y;
        covariance += centered_x * centered_y;
    }
    let denominator = n - 1.0;
    let std_x = (variance_x / denominator).sqrt();
    let std_y = (variance_y / denominator).sqrt();
    let correlation = covariance / denominator / (std_x * std_y);
    (
        mean_x as f32,
        mean_y as f32,
        std_x as f32,
        std_y as f32,
        correlation as f32,
    )
}

pub fn get_digits(number: f32, decimals: i32, n_digits: usize) -> (f32, i32) {
    let constant_part = number * 10.0_f32.powi(decimals);
    let constant_part = constant_part.floor() / 10.0_f32.powi(decimals);
    let variable_part = number - constant_part;
    let variable_part = variable_part * 10.0_f32.powi(decimals + n_digits as i32);
    (constant_part, variable_part.floor() as i32)
}

/// Compare all five statistics with a full-precision absolute tolerance.
pub fn is_error_still_ok(data: &Data, target: &Data, tolerance: f32) -> bool {
    let measured = compute_stats(data);
    let expected = compute_stats(target);
    [
        (measured.0 - expected.0).abs(),
        (measured.1 - expected.1).abs(),
        (measured.2 - expected.2).abs(),
        (measured.3 - expected.3).abs(),
        (measured.4 - expected.4).abs(),
    ]
    .into_iter()
    .all(|error| error <= tolerance)
}

pub fn max_stats_error(data: &Data, target: &Data) -> f32 {
    let measured = compute_stats(data);
    let expected = compute_stats(target);
    [
        (measured.0 - expected.0).abs(),
        (measured.1 - expected.1).abs(),
        (measured.2 - expected.2).abs(),
        (measured.3 - expected.3).abs(),
        (measured.4 - expected.4).abs(),
    ]
    .into_iter()
    .fold(0.0, f32::max)
}

pub fn min_distance_segment(point: (f32, f32), line: Line) -> f32 {
    let (start, end) = line;
    let direction = (end.0 - start.0, end.1 - start.1);
    let length_squared = direction.0 * direction.0 + direction.1 * direction.1;
    if length_squared == 0.0 {
        return ((point.0 - start.0).powi(2) + (point.1 - start.1).powi(2)).sqrt();
    }
    let projection =
        ((point.0 - start.0) * direction.0 + (point.1 - start.1) * direction.1) / length_squared;
    let t = projection.clamp(0.0, 1.0);
    let closest = (start.0 + t * direction.0, start.1 + t * direction.1);
    ((point.0 - closest.0).powi(2) + (point.1 - closest.1).powi(2)).sqrt()
}

pub fn mean_contour_distance(data: &Data, fixed_lines: &[Line]) -> f32 {
    let total = data
        .x
        .iter()
        .zip(data.y.iter())
        .map(|(&x, &y)| {
            fixed_lines
                .iter()
                .map(|line| min_distance_segment((x, y), *line))
                .fold(f32::INFINITY, f32::min)
        })
        .sum::<f32>();
    total / data.x.len() as f32
}

pub fn mean_squared_contour_distance(data: &Data, fixed_lines: &[Line]) -> f32 {
    let total = data
        .x
        .iter()
        .zip(data.y.iter())
        .map(|(&x, &y)| {
            let distance = fixed_lines
                .iter()
                .map(|line| min_distance_segment((x, y), *line))
                .fold(f32::INFINITY, f32::min);
            distance * distance
        })
        .sum::<f32>();
    total / data.x.len() as f32
}

/// One shape-objective perturbation. All random choices use the caller's RNG.
pub fn perturb_data<R: Rng + ?Sized>(
    data: &Data,
    temperature: f64,
    allowed_distance: f32,
    fixed_lines: &[Line],
    x_bounds: (f32, f32),
    y_bounds: (f32, f32),
    rng: &mut R,
) -> Data {
    assert!(
        !fixed_lines.is_empty(),
        "at least one contour segment is required"
    );
    let mut new_data = data.clone();
    let index = rng.gen_range(0..data.x.len());
    let allow_worse_objective = rng.gen_bool(temperature.clamp(0.0, 1.0));
    let old_distance = fixed_lines
        .iter()
        .map(|line| min_distance_segment((data.x[index], data.y[index]), *line))
        .fold(f32::INFINITY, f32::min);
    let normal = Normal::new(0.0, 0.1).unwrap();
    loop {
        let x = data.x[index] + rng.sample::<f32, _>(normal);
        let y = data.y[index] + rng.sample::<f32, _>(normal);
        let new_distance = fixed_lines
            .iter()
            .map(|line| min_distance_segment((x, y), *line))
            .fold(f32::INFINITY, f32::min);
        let in_bounds = x_bounds.0 <= x && x <= x_bounds.1 && y_bounds.0 <= y && y <= y_bounds.1;
        if in_bounds
            && (new_distance < old_distance
                || allow_worse_objective
                || new_distance < allowed_distance)
        {
            new_data.x[index] = x;
            new_data.y[index] = y;
            return new_data;
        }
    }
}

fn covariance(data: &Data) -> ([f64; 2], [[f64; 2]; 2]) {
    let n = data.x.len() as f64;
    let mean = [
        data.x.iter().map(|value| *value as f64).sum::<f64>() / n,
        data.y.iter().map(|value| *value as f64).sum::<f64>() / n,
    ];
    let mut covariance = [[0.0; 2]; 2];
    for (&x, &y) in data.x.iter().zip(data.y.iter()) {
        let centered = [x as f64 - mean[0], y as f64 - mean[1]];
        covariance[0][0] += centered[0] * centered[0];
        covariance[0][1] += centered[0] * centered[1];
        covariance[1][1] += centered[1] * centered[1];
    }
    let denominator = n - 1.0;
    covariance[0][0] /= denominator;
    covariance[0][1] /= denominator;
    covariance[1][0] = covariance[0][1];
    covariance[1][1] /= denominator;
    (mean, covariance)
}

fn symmetric_power(matrix: [[f64; 2]; 2], power: f64) -> Result<[[f64; 2]; 2], String> {
    let angle = 0.5 * (2.0 * matrix[0][1]).atan2(matrix[0][0] - matrix[1][1]);
    let (sine, cosine) = angle.sin_cos();
    let lambda_1 = cosine * cosine * matrix[0][0]
        + 2.0 * sine * cosine * matrix[0][1]
        + sine * sine * matrix[1][1];
    let lambda_2 = sine * sine * matrix[0][0] - 2.0 * sine * cosine * matrix[0][1]
        + cosine * cosine * matrix[1][1];
    if lambda_1 <= 1e-12 || lambda_2 <= 1e-12 || !lambda_1.is_finite() || !lambda_2.is_finite() {
        return Err(format!(
            "rank-deficient covariance eigenvalues: {lambda_1}, {lambda_2}"
        ));
    }
    let p1 = lambda_1.powf(power);
    let p2 = lambda_2.powf(power);
    Ok([
        [
            cosine * cosine * p1 + sine * sine * p2,
            sine * cosine * (p1 - p2),
        ],
        [
            sine * cosine * (p1 - p2),
            sine * sine * p1 + cosine * cosine * p2,
        ],
    ])
}

fn multiply(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn projection_pass(
    data: &Data,
    target_mean: [f64; 2],
    target_cov: [[f64; 2]; 2],
) -> Result<Data, String> {
    let (mean, source_cov) = covariance(data);
    let transform = multiply(
        symmetric_power(source_cov, -0.5)?,
        symmetric_power(target_cov, 0.5)?,
    );
    let mut result = Data {
        x: Vec::with_capacity(data.x.len()),
        y: Vec::with_capacity(data.y.len()),
    };
    for (&x, &y) in data.x.iter().zip(data.y.iter()) {
        let centered = [x as f64 - mean[0], y as f64 - mean[1]];
        result.x.push(
            (centered[0] * transform[0][0] + centered[1] * transform[1][0] + target_mean[0]) as f32,
        );
        result.y.push(
            (centered[0] * transform[0][1] + centered[1] * transform[1][1] + target_mean[1]) as f32,
        );
    }
    Ok(result)
}

/// Analytically project data to the target sample mean and covariance.
pub fn project_to_target_moments(data: &Data, target: &Data) -> Result<Data, String> {
    let (target_mean, target_cov) = covariance(target);
    let first = projection_pass(data, target_mean, target_cov)?;
    projection_pass(&first, target_mean, target_cov)
}

pub fn read_data(filename: &str) -> Data {
    let input = std::fs::read_to_string(filename).unwrap();
    let points = input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.split(',');
            let x = fields.next().unwrap().parse::<f32>().unwrap();
            let y = fields.next().unwrap().parse::<f32>().unwrap();
            (x, y)
        })
        .collect::<Vec<_>>();
    Data {
        x: points.iter().map(|point| point.0).collect(),
        y: points.iter().map(|point| point.1).collect(),
    }
}

pub fn ease_in_out_quad(t: f64) -> f64 {
    if t < 0.5 {
        2.0 * t.powi(2)
    } else {
        let value = t * 2.0 - 1.0;
        -0.5 * (value * (value - 2.0) - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn stats_include_sample_correlation() {
        let data = Data {
            x: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            y: vec![2.0, 4.0, 6.0, 8.0, 10.0],
        };
        let stats = compute_stats(&data);
        assert_eq!(stats.0, 3.0);
        assert_eq!(stats.1, 6.0);
        assert!((stats.4 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn projection_matches_all_five_target_stats() {
        let target = Data {
            x: vec![1.0, 2.0, 4.0, 7.0, 8.0, 9.0],
            y: vec![8.0, 4.0, 7.0, 2.0, 3.0, 9.0],
        };
        let candidate = Data {
            x: vec![-2.0, 0.5, 1.0, 3.5, 8.0, 12.0],
            y: vec![1.0, 7.0, -3.0, 4.0, 10.0, 2.0],
        };
        let projected = project_to_target_moments(&candidate, &target).unwrap();
        assert!(max_stats_error(&projected, &target) < 1e-5);
    }

    #[test]
    fn perturbations_are_seed_reproducible() {
        let data = Data {
            x: vec![0.0, 1.0, 2.0],
            y: vec![0.0, 1.0, 0.0],
        };
        let lines = vec![((-10.0, 0.0), (10.0, 0.0))];
        let mut first = rand::rngs::StdRng::seed_from_u64(9);
        let mut second = rand::rngs::StdRng::seed_from_u64(9);
        let a = perturb_data(
            &data,
            0.2,
            1.0,
            &lines,
            (-20.0, 20.0),
            (-20.0, 20.0),
            &mut first,
        );
        let b = perturb_data(
            &data,
            0.2,
            1.0,
            &lines,
            (-20.0, 20.0),
            (-20.0, 20.0),
            &mut second,
        );
        assert_eq!(a, b);
    }
}
