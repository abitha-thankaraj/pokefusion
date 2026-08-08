//! Runtime contour CSV loading.

use crate::types::Line;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug)]
struct ContourRow {
    order: usize,
    point: (f32, f32),
    closed: bool,
}

/// Load `contour_id,order,x,y,closed` rows and convert them to line segments.
pub fn load_contour_file(path: &Path) -> Result<Vec<Line>, String> {
    let input = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut groups: BTreeMap<usize, Vec<ContourRow>> = BTreeMap::new();
    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || (line_index == 0 && line.starts_with("contour_id,")) {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        if fields.len() != 5 {
            return Err(format!(
                "{}:{}: expected 5 columns, got {}",
                path.display(),
                line_index + 1,
                fields.len()
            ));
        }
        let parse_error = |field: &str| {
            format!(
                "{}:{}: invalid {field} in contour row",
                path.display(),
                line_index + 1
            )
        };
        let contour_id = fields[0]
            .parse::<usize>()
            .map_err(|_| parse_error("contour_id"))?;
        let order = fields[1]
            .parse::<usize>()
            .map_err(|_| parse_error("order"))?;
        let x = fields[2].parse::<f32>().map_err(|_| parse_error("x"))?;
        let y = fields[3].parse::<f32>().map_err(|_| parse_error("y"))?;
        let closed = fields[4]
            .parse::<bool>()
            .map_err(|_| parse_error("closed"))?;
        if !x.is_finite() || !y.is_finite() {
            return Err(parse_error("finite coordinate"));
        }
        groups.entry(contour_id).or_default().push(ContourRow {
            order,
            point: (x, y),
            closed,
        });
    }
    if groups.is_empty() {
        return Err(format!("{}: no contour rows", path.display()));
    }

    let mut segments = Vec::new();
    for (contour_id, mut rows) in groups {
        rows.sort_by_key(|row| row.order);
        if rows.len() < 2 {
            return Err(format!(
                "{}: contour {contour_id} has fewer than 2 vertices",
                path.display()
            ));
        }
        for (expected, row) in rows.iter().enumerate() {
            if row.order != expected {
                return Err(format!(
                    "{}: contour {contour_id} orders must be contiguous from zero",
                    path.display()
                ));
            }
            if row.closed != rows[0].closed {
                return Err(format!(
                    "{}: contour {contour_id} mixes closed values",
                    path.display()
                ));
            }
        }
        segments.extend(rows.windows(2).map(|pair| (pair[0].point, pair[1].point)));
        if rows[0].closed {
            segments.push((rows.last().unwrap().point, rows[0].point));
        }
    }
    if segments.is_empty() {
        return Err(format!("{}: no line segments", path.display()));
    }
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_closed_runtime_contour() {
        let path =
            std::env::temp_dir().join(format!("datasaurust-contour-{}.csv", std::process::id()));
        std::fs::write(
            &path,
            "contour_id,order,x,y,closed\n0,0,0,0,true\n0,1,1,0,true\n0,2,1,1,true\n",
        )
        .unwrap();
        let lines = load_contour_file(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2], ((1.0, 1.0), (0.0, 0.0)));
    }
}
