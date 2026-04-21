//! Local desktop launcher for the Solitaire Workbench.
//!
//! The binary starts a loopback-only HTTP server, opens the user's default
//! browser, and serves an embedded web UI that talks to structured Rust backend
//! commands. It intentionally does not shell out to the CLI.

mod server;
mod workbench;

use std::{error::Error, net::TcpListener};

fn main() -> Result<(), Box<dyn Error>> {
    create_default_workspace_dirs();

    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let url = format!("http://{addr}/");

    println!("Solitaire Workbench is running at {url}");
    println!("Close this window or press Ctrl+C to stop the local workbench server.");
    open_browser(&url);

    server::serve(listener)?;
    Ok(())
}

fn create_default_workspace_dirs() {
    for dir in [
        "sessions",
        "reports",
        "regression",
        "oracle",
        "data",
        "models",
    ] {
        let _ = std::fs::create_dir_all(dir);
    }
}

fn open_browser(url: &str) {
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }

    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}
