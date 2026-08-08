use clap::Parser;
use gnuplot::{AxesCommon, Caption, Color, Figure, Fix, Graph};
use kdam::tqdm;
use rand::{Rng, SeedableRng};
use rand_distr::Normal;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

use datasaurust::contour::load_contour_file;
use datasaurust::optim::*;
use datasaurust::shapes::get_shape;
use datasaurust::types::{Data, Line};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Initial headerless x,y dataset whose moments must be preserved.
    #[arg(short, long, default_value = "data/seed_datasets/Datasaurus_data.csv")]
    dataset: String,

    /// Output CSV basename.
    #[arg(short, long, default_value = "output")]
    output: String,

    /// Number of seeded contour-loss updates.
    #[arg(short, long, default_value_t = 4_000_000)]
    num_iterations: u32,

    #[arg(short, long, default_value_t = false)]
    plot: bool,

    #[arg(short, long, default_value_t = false)]
    save_plots: bool,

    #[arg(short, long, default_value_t = false, conflicts_with = "gaussian")]
    uniform: bool,

    #[arg(short, long, default_value_t = false, conflicts_with = "uniform")]
    gaussian: bool,

    #[arg(short, long, default_value_t = 10_000)]
    log_interval: u32,

    /// Display-only precision; never used for acceptance.
    #[arg(long, default_value_t = 2)]
    decimals: i32,

    /// Maximum accepted final mean distance to the contour.
    #[arg(long, default_value_t = 1.0)]
    allowed_distance: f32,

    /// Backward-compatible built-in target shape.
    #[arg(long, conflicts_with = "shape_file")]
    shape: Option<String>,

    /// Runtime contour CSV with contour_id,order,x,y,closed columns.
    #[arg(long, value_name = "PATH", conflicts_with = "shape")]
    shape_file: Option<PathBuf>,

    /// Seed shared by initialization and every optimizer random choice.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 0.0001)]
    min_temperature: f64,

    #[arg(long, default_value_t = 0.4)]
    max_temperature: f64,

    /// Absolute tolerance for all five final statistics.
    #[arg(long, default_value_t = 1e-4)]
    stats_tolerance: f32,

    /// Reproject moments after this many accepted shape moves; zero disables periodic projection.
    #[arg(long, default_value_t = 1_000)]
    project_every: u32,

    /// Optional JSON report for the final full-precision acceptance decision.
    #[arg(long, value_name = "PATH")]
    manifest_out: Option<PathBuf>,
}

fn shape_name(args: &Args) -> String {
    if let Some(path) = &args.shape_file {
        path.file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("runtime-contour")
            .to_owned()
    } else {
        args.shape.clone().unwrap_or_else(|| "cat".to_owned())
    }
}

fn load_shape(args: &Args) -> Result<Vec<Line>, String> {
    if let Some(path) = &args.shape_file {
        load_contour_file(path)
    } else {
        Ok(get_shape(args.shape.as_deref().unwrap_or("cat"), 0.0, 0.0))
    }
}

fn initialize_data(args: &Args, rng: &mut rand::rngs::StdRng) -> Data {
    if args.uniform {
        let mut data = Data {
            x: vec![0.0; 1_000],
            y: vec![0.0; 1_000],
        };
        for index in 0..data.x.len() {
            data.x[index] = rng.gen_range(20.0..80.0);
            data.y[index] = rng.gen_range(20.0..80.0);
        }
        data
    } else if args.gaussian {
        let normal_x = Normal::new(55.0, 16.0).unwrap();
        let normal_y = Normal::new(50.0, 20.0).unwrap();
        let mut data = Data {
            x: vec![0.0; 800],
            y: vec![0.0; 800],
        };
        for index in 0..data.x.len() {
            data.x[index] = rng.sample::<f32, _>(normal_x).clamp(1.0, 98.0);
            data.y[index] = rng.sample::<f32, _>(normal_y).clamp(1.0, 98.0);
        }
        data
    } else {
        read_data(&args.dataset)
    }
}

fn save_plot(
    figure: &mut Figure,
    data: &Data,
    stats: (f32, f32, f32, f32, f32),
    bounds: ((f32, f32), (f32, f32)),
    decimals: usize,
    destination: Option<&Path>,
) {
    figure.clear_axes();
    let labels = [
        format!("X Mean: {:.decimals$}", stats.0),
        format!("Y Mean: {:.decimals$}", stats.1),
        format!("X SD:   {:.decimals$}", stats.2),
        format!("Y SD:   {:.decimals$}", stats.3),
        format!("Corr:   {:.decimals$}", stats.4),
    ];
    let axes = figure
        .axes2d()
        .set_title("DatasauRust", &[])
        .set_x_label("X", &[])
        .set_y_label("Y", &[])
        .set_x_range(Fix(bounds.0 .0 as f64), Fix(bounds.0 .1 as f64))
        .set_y_range(Fix(bounds.1 .0 as f64), Fix(bounds.1 .1 as f64))
        .points(
            data.x.iter(),
            data.y.iter(),
            &[
                Caption(""),
                gnuplot::PointSymbol('O'),
                gnuplot::PointSize(1.5),
                Color("black"),
            ],
        );
    for (index, label) in labels.iter().enumerate() {
        axes.label(
            label,
            Graph(0.32),
            Graph(0.95 - index as f64 * 0.055),
            &[
                gnuplot::Font("Monospace", 14.0),
                gnuplot::TextColor("black"),
            ],
        );
    }
    if let Some(path) = destination {
        figure.save_to_png(path, 640, 480).unwrap();
    } else {
        figure.show_and_keep_running().unwrap();
    }
}

fn stats_json(stats: (f32, f32, f32, f32, f32)) -> serde_json::Value {
    json!({
        "mean_x": stats.0,
        "mean_y": stats.1,
        "std_x": stats.2,
        "std_y": stats.3,
        "corr": stats.4,
    })
}

fn main() {
    let args = Args::parse();
    assert!(
        args.stats_tolerance >= 0.0,
        "--stats-tolerance must be non-negative"
    );
    assert!(args.log_interval > 0, "--log-interval must be positive");
    let fixed_lines = load_shape(&args).unwrap_or_else(|error| panic!("{error}"));
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let initial_data = initialize_data(&args, &mut rng);
    let mut best_data = initial_data.clone();
    let x_bounds = (-20.0, 130.0);
    let y_bounds = (-10.0, 145.0);
    let label = shape_name(&args);
    let log_folder = PathBuf::from("logs").join(&label);
    std::fs::create_dir_all(&log_folder).unwrap();
    let mut figure = Figure::new();

    for iteration in tqdm!(0..args.num_iterations) {
        let temperature = args.min_temperature
            + (args.max_temperature - args.min_temperature)
                * ease_in_out_quad(
                    (args.num_iterations - iteration) as f64 / args.num_iterations as f64,
                );
        best_data = perturb_data(
            &best_data,
            temperature,
            args.allowed_distance,
            &fixed_lines,
            x_bounds,
            y_bounds,
            &mut rng,
        );
        let accepted = iteration + 1;
        if args.project_every > 0 && accepted % args.project_every == 0 {
            best_data = project_to_target_moments(&best_data, &initial_data)
                .unwrap_or_else(|error| panic!("moment projection failed: {error}"));
        }
        if (args.plot || args.save_plots) && iteration % args.log_interval == 0 {
            let destination = args
                .save_plots
                .then(|| log_folder.join(format!("{:06}.png", iteration / args.log_interval)));
            save_plot(
                &mut figure,
                &best_data,
                compute_stats(&best_data),
                (x_bounds, y_bounds),
                args.decimals.max(0) as usize,
                destination.as_deref(),
            );
        }
    }

    best_data = project_to_target_moments(&best_data, &initial_data)
        .unwrap_or_else(|error| panic!("final moment projection failed: {error}"));
    let measured_stats = compute_stats(&best_data);
    let target_stats = compute_stats(&initial_data);
    let stats_error = max_stats_error(&best_data, &initial_data);
    if !is_error_still_ok(&best_data, &initial_data, args.stats_tolerance) {
        panic!(
            "final statistics error {stats_error:e} exceeds tolerance {:e}",
            args.stats_tolerance
        );
    }
    let contour_distance = mean_contour_distance(&best_data, &fixed_lines);
    let contour_mse = mean_squared_contour_distance(&best_data, &fixed_lines);
    if contour_distance > args.allowed_distance {
        panic!(
            "final mean contour distance {contour_distance} exceeds threshold {}",
            args.allowed_distance
        );
    }

    let output_path = log_folder.join(format!("{}.csv", args.output));
    let mut output = std::fs::File::create(&output_path).unwrap();
    for (&x, &y) in best_data.x.iter().zip(best_data.y.iter()) {
        writeln!(output, "{x},{y}").unwrap();
    }

    if let Some(manifest_path) = &args.manifest_out {
        if let Some(parent) = manifest_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let manifest = json!({
            "schema_version": 1,
            "status": "accepted",
            "seed": args.seed,
            "iterations": args.num_iterations,
            "shape": {
                "built_in": args.shape.as_deref().or(if args.shape_file.is_none() { Some("cat") } else { None }),
                "file": args.shape_file,
                "segments": fixed_lines.len(),
            },
            "points": {"path": output_path, "count": best_data.x.len()},
            "target_stats": stats_json(target_stats),
            "measured_stats": stats_json(measured_stats),
            "max_stats_error": stats_error,
            "stats_tolerance": args.stats_tolerance,
            "project_every": args.project_every,
            "shape_metrics": {
                "mean_contour_distance": contour_distance,
                "mean_squared_contour_distance": contour_mse,
            },
        });
        std::fs::write(
            manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap() + "\n",
        )
        .unwrap();
    }

    println!(
        "accepted {} points: max stats error={stats_error:e}, mean contour distance={contour_distance:.6}",
        best_data.x.len()
    );
}
