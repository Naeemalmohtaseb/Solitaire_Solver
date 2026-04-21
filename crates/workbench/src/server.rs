//! Minimal loopback HTTP server for the embedded Workbench UI.

use std::{
    collections::HashMap,
    error::Error,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{Arc, Mutex},
    thread,
};

use serde::Serialize;
use serde_json::{json, Value};

use crate::workbench::{self, TaskStore};

const INDEX_HTML: &str = include_str!("assets/index.html");
const APP_JS: &str = include_str!("assets/app.js");
const STYLES_CSS: &str = include_str!("assets/styles.css");

/// Runs the local workbench server until the process exits.
pub fn serve(listener: TcpListener) -> Result<(), Box<dyn Error>> {
    let tasks = TaskStore::default();
    let tasks = Arc::new(Mutex::new(tasks));

    for stream in listener.incoming() {
        let stream = stream?;
        let tasks = Arc::clone(&tasks);
        thread::spawn(move || {
            if let Err(err) = handle_connection(stream, tasks) {
                eprintln!("workbench request failed: {err}");
            }
        });
    }
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    tasks: Arc<Mutex<TaskStore>>,
) -> Result<(), Box<dyn Error>> {
    let request = HttpRequest::read(&mut stream)?;
    let response = route_request(request, tasks);
    stream.write_all(&response.into_bytes())?;
    Ok(())
}

fn route_request(request: HttpRequest, tasks: Arc<Mutex<TaskStore>>) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => {
            HttpResponse::ok("text/html; charset=utf-8", INDEX_HTML)
        }
        ("GET", "/app.js") => HttpResponse::ok("application/javascript; charset=utf-8", APP_JS),
        ("GET", "/styles.css") => HttpResponse::ok("text/css; charset=utf-8", STYLES_CSS),
        ("GET", "/api/status") => json_response(workbench::status()),
        ("GET", "/api/presets") => json_response(workbench::presets()),
        ("GET", path) if path.starts_with("/api/task/") => {
            let task_id = path.trim_start_matches("/api/task/");
            json_result(workbench::task_snapshot(&tasks, task_id))
        }
        ("POST", "/api/session/parse") => json_result(workbench::parse_session(&request.body)),
        ("POST", "/api/session/load_path") => {
            json_result(workbench::load_session_from_path(&request.body))
        }
        ("POST", "/api/session/save_path") => {
            json_result(workbench::save_session_to_path(&request.body))
        }
        ("POST", "/api/game/generate") => json_result(workbench::generate_game(&request.body)),
        ("POST", "/api/game/generate_autoplay") => {
            json_result(workbench::generate_autoplay(&request.body))
        }
        ("POST", "/api/diagnostics/analyze_session") => {
            json_result(workbench::analyze_session(&request.body))
        }
        ("POST", "/api/diagnostics/analyze_current") => {
            json_result(workbench::analyze_current(&request.body))
        }
        ("POST", "/api/recommend") => json_result(workbench::recommend_move(&request.body)),
        ("POST", "/api/solve/one_step") => json_result(workbench::solve_one_step(&request.body)),
        ("POST", "/api/solve/run_to_end") => {
            json_result(workbench::solve_run_to_end(&request.body))
        }
        ("POST", "/api/autoplay/run") => json_result(workbench::run_autoplay(&request.body)),
        ("POST", "/api/benchmark/start") => {
            json_result(workbench::start_benchmark_task(&tasks, &request.body))
        }
        _ => json_error(404, "not found"),
    }
}

fn json_response(value: impl Serialize) -> HttpResponse {
    match serde_json::to_string(&value) {
        Ok(body) => HttpResponse::ok("application/json; charset=utf-8", &body),
        Err(err) => json_error(500, &format!("serialization error: {err}")),
    }
}

fn json_result(value: Result<impl Serialize, String>) -> HttpResponse {
    match value {
        Ok(value) => json_response(json!({ "ok": true, "data": value })),
        Err(err) => json_error(400, &err),
    }
}

fn json_error(status: u16, message: &str) -> HttpResponse {
    json_response_with_status(status, json!({ "ok": false, "error": message }))
}

fn json_response_with_status(status: u16, value: Value) -> HttpResponse {
    match serde_json::to_string(&value) {
        Ok(body) => HttpResponse {
            status,
            content_type: "application/json; charset=utf-8".to_string(),
            body,
        },
        Err(_) => HttpResponse {
            status,
            content_type: "application/json; charset=utf-8".to_string(),
            body: "{\"ok\":false,\"error\":\"serialization failed\"}".to_string(),
        },
    }
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

impl HttpRequest {
    fn read(stream: &mut TcpStream) -> Result<Self, Box<dyn Error>> {
        let mut buffer = Vec::new();
        let mut temp = [0u8; 4096];
        let header_end = loop {
            let count = stream.read(&mut temp)?;
            if count == 0 {
                return Err("empty request".into());
            }
            buffer.extend_from_slice(&temp[..count]);
            if let Some(index) = find_header_end(&buffer) {
                break index;
            }
            if buffer.len() > 1024 * 1024 {
                return Err("request headers too large".into());
            }
        };

        let headers = String::from_utf8_lossy(&buffer[..header_end]);
        let mut lines = headers.lines();
        let request_line = lines.next().ok_or("missing request line")?;
        let mut parts = request_line.split_whitespace();
        let method = parts.next().ok_or("missing method")?.to_string();
        let raw_path = parts.next().ok_or("missing path")?;
        let path = raw_path.split('?').next().unwrap_or(raw_path).to_string();

        let mut header_map = HashMap::new();
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                header_map.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
            }
        }
        let content_length = header_map
            .get("content-length")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);

        let body_start = header_end + 4;
        while buffer.len().saturating_sub(body_start) < content_length {
            let count = stream.read(&mut temp)?;
            if count == 0 {
                break;
            }
            buffer.extend_from_slice(&temp[..count]);
        }
        let body = buffer
            .get(body_start..body_start + content_length)
            .unwrap_or_default()
            .to_vec();

        Ok(Self { method, path, body })
    }
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|window| window == b"\r\n\r\n")
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    content_type: String,
    body: String,
}

impl HttpResponse {
    fn ok(content_type: &str, body: &str) -> Self {
        Self {
            status: 200,
            content_type: content_type.to_string(),
            body: body.to_string(),
        }
    }

    fn into_bytes(self) -> Vec<u8> {
        let status_text = match self.status {
            200 => "OK",
            400 => "Bad Request",
            404 => "Not Found",
            500 => "Internal Server Error",
            _ => "OK",
        };
        let response = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{}",
            self.status,
            status_text,
            self.content_type,
            self.body.as_bytes().len(),
            self.body
        );
        response.into_bytes()
    }
}
